import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_dataset(dataset_name, test_file_path, model, tokenizer):
    """
    Loads a dataset, runs predictions, and calculates/saves metrics.
    """
    print(f"\n--- Starting Evaluation for: {dataset_name.upper()} ---")

    try:
        test_dataset = load_dataset("json", data_files={"test": test_file_path})['test']
    except Exception as e:
        print(f"Could not load or process file {test_file_path}. Error: {e}")
        return

    processed_test_dataset = test_dataset.rename_column("exist", "label")
    columns_to_remove = ["pragma", "private", "reduction", "path"]
    processed_test_dataset = processed_test_dataset.remove_columns(columns_to_remove)

    def preprocess_function(examples):
        return tokenizer(examples['code'], truncation=True, padding='max_length', max_length=512)

    print(f"Tokenizing {dataset_name} test dataset...")
    tokenized_test = processed_test_dataset.map(preprocess_function, batched=True)

    eval_output_dir = os.path.join("results", f"eval_output_{dataset_name}")
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=eval_output_dir, report_to="none"),
    )

    print("Making predictions...")
    predictions = trainer.predict(tokenized_test)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    print(f"\n--- Performance Report for {dataset_name.upper()} ---")
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Parallel', 'Parallel'])
    disp.plot()
    plot_save_path = os.path.join("results", f"confusion_matrix_{dataset_name}.png")
    plt.savefig(plot_save_path)
    plt.close()
    print(f"\nConfusion matrix saved to {plot_save_path}")
    print(f"--- Evaluation Complete for: {dataset_name.upper()} ---")


if __name__ == "__main__":
    print("--- Initializing Benchmark Evaluation Script ---")
    model_path = "results/best_model"

    print(f"Loading fine-tuned model from: {model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except OSError:
        print(f"ERROR: Model not found at {model_path}.")
        print("Please run train.py first to fine-tune and save the model.")
        exit()

    test_files = {
        "original": "data/test.jsonl",
        "nas": "data/test_nas.jsonl",
        "poly": "data/test_poly.jsonl",
        "spec": "data/test_spec.jsonl"
    }

    for name, filepath in test_files.items():
        if os.path.exists(filepath):
            evaluate_dataset(name, filepath, model, tokenizer)
        else:
            print(f"\n--- WARNING: File not found, skipping. Path: {filepath} ---")

    print("\n--- All Benchmark Evaluations Complete ---")