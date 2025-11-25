**SemEval 2026 Task 5 — Ambiguous Story Plausibility Rating**

This repository contains a complete implementation for SemEval 2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Stories through Narrative Understanding.
The goal of this task is to predict how plausible (1–5) a particular word sense is when used in a short, ambiguous 5-sentence narrative.
Our system is built using RoBERTa-base with a regression head, trained on the official dataset, and evaluated using the official SemEval scoring script.

**Task Overview**

SemEval Task 5 introduces the AmbiStory dataset, where each example contains:
3 Precontext sentences
1 Ambiguous sentence containing a homonym
1 Optional ending
2 word senses with human plausibility ratings
Mean and Standard Deviation of human annotations

The model must output a continuous plausibility score from 1 to 5, representing how plausible the meaning is in the narrative.


**Installation**

***1. Clone the repository***

git clone https://github.com/tanvinimbalkar/SemEval-2026-Task-5-Ambiguous-Story-Plausibility-Rating.git

cd Sem_Eval_Task-05

***2. Install dependencies***

pip install -r requirements.txt

**Training the Model**

Run the training script:

python src/train_model.py


This will:

Load the RoBERTa tokenizer & base model

Process train.json and dev.json from the dataset

Train for 3 epochs

Save model config + tokenizer files to:

models/semeval_roberta/


**Generating Predictions (Dev Set)**
python src/generate_solution_from_dev.py

This generates:
predictions_dev.jsonl


Example line:
{"id": "0", "pred": 4.0}


**Formatting Predictions for Scoring**

SemEval requires:
{"id": "0", "prediction": 4.0}

Convert your predictions:
python src/fix_predictions.py

Output:
predictions_dev_fixed.jsonl

**Evaluating Your Model (Official SemEval Script)**

python semeval26-05-scripts/scoring.py \
    semeval26-05-scripts/input/ref/solution.jsonl \
    predictions_dev_fixed.jsonl \
    scores.json

Example Output:
{
  "spearman": 0.319,
  "within_std_accuracy": 0.474
}


Spearman → ranking correlation between human scores & model
Within-Std Accuracy → prediction is within ±1 std of human mean


**Example Story (Illustration)**
Precontext: The detectives arrived at the abandoned train station...
Ambiguous Sentence: They followed the track.
Ending: They began to run along the abandoned railway line...
Homonym: track
Judged Meaning: "a pair of parallel rails..."
Average Human Rating: 3.6


Model output:
prediction = 4.0


**Requirements**
torch
transformers
datasets
numpy
scipy
pandas
tqdm

**Notes**

Model weights (.bin, .safetensors) are not included due to GitHub size limits/n
The repository includes the official scoring scripts for reproducibility.

The notebook (semeval-task05.ipynb) contains the complete training pipeline used on Kaggle.

Prediction file format strictly follows SemEval guidelines.


give in this form i guesslatex
