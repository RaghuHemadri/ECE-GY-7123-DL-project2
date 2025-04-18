import logging
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    RobertaModel,
    RobertaPreTrainedModel,
    AutoConfig
)
from datasets import load_dataset, Dataset, ClassLabel
import evaluate
import numpy as np
from peft import LoraConfig, get_peft_model
from torch import nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import nlpaug.augmenter.word as naw
import torch

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration
# ---------------------------
class Config:
    # Configuration for the training process
    base_model = "roberta-base"  # Base model to use
    output_dir = "results_lora"  # Directory to save results
    use_fnn = False  # Whether to use a custom feedforward neural network
    use_augmentation = False  # Whether to use data augmentation
    use_early_stopping = True  # Whether to use early stopping
    use_weight_decay = False  # Whether to apply weight decay
    freeze_base_model = True  # Whether to freeze base model parameters
    use_mc_dropout_inference = False  # Whether to use MC Dropout during inference
    early_stopping_patience = 3  # Patience for early stopping
    weight_decay_value = 0.01  # Weight decay value
    mc_dropout_iterations = 10  # Number of MC Dropout iterations
    train_last_k_layers = 2  # Number of last layers to train
    max_seq_length = 128  # Maximum sequence length
    train_batch_size = 32  # Training batch size
    eval_batch_size = 64  # Evaluation batch size
    num_train_epochs = 1  # Number of training epochs
    learning_rate = 5e-6  # Learning rate

    # LoRA Configuration
    lora_r = 2  # LoRA rank
    lora_alpha = 4  # LoRA alpha
    lora_dropout = 0.05  # LoRA dropout rate
    lora_bias = "none"  # LoRA bias type
    lora_target_modules = ["query", "value"]  # Target modules for LoRA
    lora_task_type = "SEQ_CLS"  # Task type for LoRA

# ---------------------------
# Custom Model Class
# ---------------------------
class RobertaWithFNN(RobertaPreTrainedModel):
    # Custom model class with an additional feedforward neural network
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)  # Base Roberta model
        self.classifier = nn.Sequential(  # Custom feedforward neural network
            nn.Linear(config.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.num_labels)
        )
        self.init_weights()
    
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Forward pass
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use CLS token representation
        logits = self.classifier(pooled_output)  # Pass through classifier
        loss = None
        if labels is not None:
            # Compute loss with label smoothing
            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# ---------------------------
# Utility Functions
# ---------------------------
def preprocess_data(tokenizer, dataset, max_length):
    # Preprocess dataset by tokenizing text
    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")
    return dataset.map(preprocess, batched=True, remove_columns=["text"])

def augment_dataset(dataset):
    # Augment dataset using synonym replacement
    aug = naw.SynonymAug(aug_src='wordnet')
    def augment_text(example):
        try:
            augmented = aug.augment(example["text"], n=1)
            return {"text": augmented[0] if isinstance(augmented, list) else augmented}
        except Exception:
            return {"text": example["text"]}
    subset = dataset["train"]
    augmented = subset.map(augment_text)
    combined = Dataset.from_list(subset.to_list() + augmented.to_list())
    return combined, dataset["test"]

def compute_metrics(p):
    # Compute accuracy metric
    metric = evaluate.load("accuracy")
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def mc_dropout_predict(model, dataset, data_collator, device, iterations):
    # Perform MC Dropout inference
    model.train()  # Enable dropout during inference
    loader = DataLoader(dataset, batch_size=64, collate_fn=data_collator)
    all_logits = []
    for _ in range(iterations):
        iteration_logits = []
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            iteration_logits.append(outputs.logits.cpu().numpy())
        all_logits.append(np.concatenate(iteration_logits, axis=0))
    mean_logits = np.mean(np.array(all_logits), axis=0)
    return np.argmax(mean_logits, axis=1)

def freeze_model_parameters(model):
    # Freeze parameters of the base model
    logger.info("Freezing base model parameters")
    for name, param in model.named_parameters():
        if "lora" not in name and "classifier" not in name:
            param.requires_grad = False

# ---------------------------
# Main Training Function
# ---------------------------
def train_model(config):
    # Main function to train the model
    logger.info("Loading tokenizer and dataset")
    tokenizer = RobertaTokenizer.from_pretrained(config.base_model)
    tokenizer.model_max_length = config.max_seq_length

    # Load and optionally augment dataset
    if config.use_augmentation:
        train_dataset, test_dataset = augment_dataset(load_dataset("ag_news"))
    else:
        dataset = load_dataset("ag_news")
        train_dataset, test_dataset = dataset["train"], dataset["test"]

    # Preprocess datasets
    tokenized_train_dataset = preprocess_data(tokenizer, train_dataset, config.max_seq_length)
    tokenized_test_dataset = preprocess_data(tokenizer, test_dataset, config.max_seq_length)
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")

    # Prepare label mappings
    num_labels = len(set(tokenized_train_dataset["labels"]))
    label_names = tokenized_train_dataset.features["labels"].names if isinstance(tokenized_train_dataset.features["labels"], ClassLabel) else ["World", "Sports", "Business", "Sci/Tech"]
    id2label = {i: name for i, name in enumerate(label_names)}
    label2id = {name: i for i, name in enumerate(label_names)}

    # Initialize model
    if config.use_fnn:
        model_config = AutoConfig.from_pretrained(config.base_model, num_labels=num_labels)
        model = RobertaWithFNN.from_pretrained(config.base_model, config=model_config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(config.base_model, num_labels=num_labels, id2label=id2label, label2id=label2id)

    # Log number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of actual parameters: {trainable_params}")

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        target_modules=config.lora_target_modules,
        task_type=config.lora_task_type
    )
    model = get_peft_model(model, lora_config)

    # Freeze base model parameters if required
    if config.freeze_base_model:
        freeze_model_parameters(model)

    # Log number of trainable parameters after freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {trainable_params}")

    # Ensure trainable parameters are below a threshold
    assert trainable_params < 1_000_000, f"Trainable parameters exceed 1 million: {trainable_params}"

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f'./trained_models/{config.output_dir}',
        evaluation_strategy='steps',
        save_strategy='steps',
        eval_steps=500,
        save_steps=4000,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay_value if config.use_weight_decay else 0.0,
        logging_dir='./logs',
        logging_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        report_to="wandb",
    )

    # Initialize data collator and callbacks
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    callbacks = [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)] if config.use_early_stopping else []

    logger.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=callbacks
    )

    # Start training
    logger.info("Starting training")
    trainer.train()

    # Evaluate the model
    logger.info("Evaluating the model")
    if config.use_mc_dropout_inference:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        predictions = mc_dropout_predict(model, tokenized_test_dataset, data_collator, device, config.mc_dropout_iterations)
        true_labels = np.array(tokenized_test_dataset["labels"])
        mc_accuracy = accuracy_score(true_labels, predictions)
        logger.info(f"MC Dropout Test Accuracy: {mc_accuracy:.4f}")
    else:
        results = trainer.evaluate()
        logger.info(f"Evaluation results: {results}")

    # Save the model and tokenizer
    logger.info("Saving the model and tokenizer")
    model.save_pretrained(f'./trained_models/{config.output_dir}/final_model')
    tokenizer.save_pretrained(f'./trained_models/{config.output_dir}/final_model')

    logger.info("Script finished successfully")

if __name__ == "__main__":
    config = Config()
    train_model(config)
