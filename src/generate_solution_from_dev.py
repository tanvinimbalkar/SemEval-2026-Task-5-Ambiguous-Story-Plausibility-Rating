import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dataset import load_dataset, LABEL_FIELD
import torch

MODEL_PATH = "/kaggle/working/models/semeval_roberta"   # path after training
DEV_PATH = "/kaggle/input/semeval26-task5-dataa/dev.json"
OUT_FILE = "/kaggle/working/predictions_dev.jsonl"

def main():
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=1, problem_type="regression"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading dev set...")
    dev = load_dataset(DEV_PATH)

    preds = []

    print("Generating predictions...")
    for ex in dev:
        inputs = tokenizer(
            ex["text"],
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**inputs)
            score = output.logits.squeeze().item()

        # Clip + round to valid SemEval range 1–5
        score = float(np.clip(round(score), 1, 5))

        preds.append({"id": ex["id"], "pred": score})

    print(f"Writing to {OUT_FILE} ...")
    with open(OUT_FILE, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")

    print("Done! File saved:", OUT_FILE)

if __name__ == "__main__":
    main()
