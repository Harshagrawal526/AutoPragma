import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- 1. SETUP PATHS AND LOAD MODEL ---
print("--- Starting Evaluation ---")
model_path = "results/best_model"
test_file = "data/test.jsonl"
output_dir = "results"

print(f"Loading fine-tuned model from: {model_path}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
except OSError:
    print(f"ERROR: Model not found at {model_path}.")
    print("Please run train.py first to fine-tune and save the model.")
    exit()

# --- 2. LOAD AND PREPARE THE TEST DATASET ---
print(f"Loading test dataset from: {test_file}")
test_dataset = load_dataset("json", data_files={"test": test_file})['test']

processed_test_dataset = test_dataset.rename_column("exist", "label")
columns_to_remove = ["pragma", "private", "reduction", "path"]
processed_test_dataset = processed_test_dataset.remove_columns(columns_to_remove)

def preprocess_function(examples):
    return tokenizer(examples['code'], truncation=True, padding='max_length', max_length=512)

print("Tokenizing test dataset...")
tokenized_test = processed_test_dataset.map(preprocess_function, batched=True)

# --- 3. MAKE PREDICTIONS ---
eval_output_dir = os.path.join(output_dir, "eval_output")
trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir=eval_output_dir, report_to="none"),
)

print("Making predictions on the test set...")
predictions = trainer.predict(tokenized_test)
predicted_labels = np.argmax(predictions.predictions, axis=1)
true_labels = predictions.label_ids

# --- 4. CALCULATE AND DISPLAY FINAL METRICS ---
print("\n--- Final Performance Report ---")
accuracy = accuracy_score(true_labels, predicted_labels)
precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nConfusion Matrix:")
cm = confusion_matrix(true_labels, predicted_labels)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Parallel', 'Parallel'])
disp.plot()
plot_save_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(plot_save_path)
print(f"\nConfusion matrix saved to {plot_save_path}")

print("\n--- Evaluation Complete ---")