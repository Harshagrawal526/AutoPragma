import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --- 1. SETUP PATHS AND LOAD THE FINE-TUNED MODEL ---
print("--- Initializing Demo Script ---")
model_path = "results/best_model"

print(f"Loading fine-tuned model from: {model_path}")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
except OSError:
    print(f"\nERROR: Model not found at {model_path}.")
    print("Please run train.py first to fine-tune and save the model.")
    exit()

# --- 2. CREATE A PREDICTION PIPELINE ---
print("\nCreating prediction pipeline...")
pragma_classifier = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1, # Use GPU if available
    top_k=None # Return both labels
)

# --- 3. DEFINE EXAMPLE CODE SNIPPETS ---
examples = [
    {
        "name": "Example 1: Parallelizable Loop (Independent Operations)",
        "code": "for (int i = 0; i < N; i++) { c[i] = a[i] + b[i]; }"
    },
    {
        "name": "Example 2: Not Parallelizable (Loop-Carried Dependency)",
        "code": "for (int i = 1; i < N; i++) { a[i] = a[i-1] + 5; }"
    },
    {
        "name": "Example 3: Parallelizable Loop (Reduction)",
        "code": "double sum = 0.0; for (int i = 0; i < N; i++) { sum += data[i]; }"
    }
]

# --- 4. RUN AND DISPLAY PREDICTIONS ---
print("\n--- Running Predictions on Hard-Coded Examples ---")

for example in examples:
    print(f"\n{'='*20} {example['name']} {'='*20}")
    print("Code Snippet:")
    print(example['code'])

    # Get the prediction from the pipeline
    results = pragma_classifier(example['code'])[0]

    # Find the label with the highest score
    best_result = max(results, key=lambda x: x['score'])
    predicted_label_name = model.config.id2label[int(best_result['label'].split('_')[-1])]
    confidence_score = best_result['score']

    print("\nModel Prediction:")
    if predicted_label_name == "LABEL_1":
        print(f"--> Result: PARALLELIZABLE (Confidence: {confidence_score:.2%})")
    else:
        print(f"--> Result: NOT PARALLELIZABLE (Confidence: {confidence_score:.2%})")

print(f"\n{'='*20} Demo Complete {'='*20}")