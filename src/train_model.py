import os
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from scipy.stats import spearmanr
from datasets import DatasetDict
from dataset import load_dataset, LABEL_FIELD

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Your offline model path (correct)
MODEL_NAME = "/kaggle/input/roberta-base-model/other/default/1/roberta-base"

OUT_DIR = "/kaggle/working/models/semeval_roberta"
MAX_LEN = 256


def tokenize(batch, tokenizer):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    enc["labels"] = batch[LABEL_FIELD]
    return enc


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds[:, 0] if preds.ndim > 1 else preds
    preds = np.clip(np.rint(preds), 1, 5)
    s = spearmanr(preds, labels).correlation
    return {"spearman": float(0 if np.isnan(s) else s)}


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train = load_dataset("/kaggle/input/semeval26-task5-dataa/train.json")
    dev = load_dataset("/kaggle/input/semeval26-task5-dataa/dev.json")

    tokenized = DatasetDict({
        "train": train.map(lambda b: tokenize(b, tokenizer), batched=True),
        "dev": dev.map(lambda b: tokenize(b, tokenizer), batched=True)
    })

    # Load model but ignore mismatched classifier weights
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True
    )

    # Kaggle-safe TrainingArguments (older transformers compatible)
    args = TrainingArguments(
        output_dir=OUT_DIR,
        do_train=True,
        do_eval=True,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=500,
        report_to=[]  # disable WandB
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)


if __name__ == "__main__":
    main()
