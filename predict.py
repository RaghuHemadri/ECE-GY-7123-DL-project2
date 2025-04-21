import logging
import torch
import pickle
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_DIR = "best_model"  # Directory containing the fine-tuned model
BATCH_SIZE = 64  # Batch size for DataLoader
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU


def load_model_and_tokenizer(model_dir: str):
    """
    Load the tokenizer and model from the specified directory.
    """
    logger.info("Loading tokenizer and model from fine-tuned checkpoint")
    tokenizer = RobertaTokenizer.from_pretrained(f"trained_models/{model_dir}/final_model")  # Load tokenizer
    model = RobertaForSequenceClassification.from_pretrained(
        f"trained_models/{model_dir}/final_model", num_labels=4  # Load model with 4 output labels
    )
    model.to(DEVICE)  # Move model to the appropriate device (CPU/GPU)
    model.eval()  # Set model to evaluation mode
    return tokenizer, model


def load_test_dataset(file_path: str) -> Dataset:
    """
    Load the custom test dataset from a pickle file.
    """
    logger.info("Loading custom test dataset")
    with open(file_path, "rb") as f:
        dataset = pickle.load(f)  # Load the dataset (expected to be a HuggingFace Dataset object)
    return dataset


def preprocess_dataset(dataset: Dataset, tokenizer: RobertaTokenizer) -> Dataset:
    """
    Tokenize the dataset with padding and truncation.
    """
    logger.info("Tokenizing custom test dataset")
    tokenizer.model_max_length = 128  # Set maximum token length for the tokenizer

    def preprocess(examples):
        # Tokenize the text with truncation and padding
        return tokenizer(
            examples["text"], truncation=True, max_length=128, padding="max_length"
        )

    # Apply the preprocessing function to the dataset
    return dataset.map(preprocess, batched=True, remove_columns=["text"])


def predict(
    model: RobertaForSequenceClassification,
    dataloader: DataLoader,
) -> torch.Tensor:
    """
    Perform predictions on the dataset using the model.
    """
    logger.info("Running predictions on the dataset")
    all_predictions = []  # List to store predictions
    for batch in tqdm(dataloader, desc="Evaluating"):  # Iterate through batches
        batch = {k: v.to(DEVICE) for k, v in batch.items()}  # Move batch to the appropriate device
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model(**batch)  # Forward pass
        preds = outputs.logits.argmax(dim=-1)  # Get predicted class labels
        all_predictions.append(preds.cpu())  # Move predictions to CPU and store
    return torch.cat(all_predictions, dim=0)  # Concatenate all predictions


def save_predictions(predictions: torch.Tensor, output_file: str):
    """
    Save predictions to a CSV file.
    """
    logger.info("Saving predictions to CSV")
    ids = list(range(len(predictions)))  # Generate IDs for each prediction
    # Create a list of dictionaries with ID and Label
    output = [{"ID": id_, "Label": label} for id_, label in zip(ids, predictions.numpy())]
    output_df = pd.DataFrame(output)  # Convert to a pandas DataFrame
    output_df.to_csv(output_file, index=False)  # Save DataFrame to a CSV file
    logger.info(f"Predictions saved to {output_file}")


def main():
    """
    Main function to load the model, preprocess the dataset, perform predictions, and save results.
    """
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)

    # Load and preprocess the test dataset
    test_dataset = load_test_dataset("test_unlabelled.pkl")  # Load test dataset from a pickle file
    tokenized_dataset = preprocess_dataset(test_dataset, tokenizer)  # Tokenize the dataset

    # Create DataLoader
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")  # Collator for padding
    dataloader = DataLoader(
        tokenized_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator  # Create DataLoader
    )

    # Perform predictions
    predictions = predict(model, dataloader)  # Run predictions on the dataset

    # Save predictions to a CSV file
    save_predictions(predictions, "predictions.csv")  # Save predictions to a file


if __name__ == "__main__":
    main()  # Execute the main function
