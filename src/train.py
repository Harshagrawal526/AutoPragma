import os
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Assume the script is run from the project root directory
# Define relative paths
data_files = {
    "train": "data/train.jsonl",
    "validation": "data/valid.jsonl",
    "test": "data/test.jsonl"
}
output_dir = "results"
best_model_dir = os.path.join(output_dir, "best_model")

# 1. Load the dataset from your JSONL files
print(f"Loading dataset from files: {data_files}")
dataset = load_dataset("json", data_files=data_files)

# 2. Prepare the labels and remove unused columns
print("Preparing labels and cleaning up the dataset...")
processed_dataset = dataset.rename_column("exist", "label")
columns_to_remove = ["pragma", "private", "reduction", "path"]
processed_dataset = processed_dataset.remove_columns(columns_to_remove)

# 3. Model Setup
model_name = "microsoft/deberta-v3-small"
print(f"Loading tokenizer and model from Hugging Face Hub: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 4. Preprocessing function to tokenize the 'code' field
def preprocess_function(examples):
    return tokenizer(examples['code'], truncation=True, padding='max_length', max_length=512)

print("Tokenizing datasets...")
tokenized_train = processed_dataset['train'].map(preprocess_function, batched=True)
tokenized_val = processed_dataset['validation'].map(preprocess_function, batched=True)

# 5. Define Metrics for Evaluation
print("Defining metrics...")
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average='binary')
    recall = recall_metric.compute(predictions=predictions, references=labels, average='binary')
    return {
        "accuracy": accuracy['accuracy'],
        "precision": precision['precision'],
        "recall": recall['recall']
    }

# 6. Define Training Arguments
print("Setting training arguments...")
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=50,
    report_to="none",
)

# 7. Create the Trainer
print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# 8. Start Fine-Tuning
print("--- Starting Training ---")
trainer.train()

# 9. Save the final best model and tokenizer
print(f"--- Saving the best model to {best_model_dir} ---")
trainer.save_model(best_model_dir)
tokenizer.save_pretrained(best_model_dir)

print("--- Training Complete ---")