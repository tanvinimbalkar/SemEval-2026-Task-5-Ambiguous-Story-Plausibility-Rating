\section*{SemEval 2026 Task 5 --- Ambiguous Story Plausibility Rating}

This repository contains a complete implementation for \textbf{SemEval 2026 Task 5: Rating Plausibility of Word Senses in Ambiguous Stories through Narrative Understanding}. 
The goal of this task is to predict how plausible (1--5) a particular word sense is when used in a short, ambiguous five-sentence narrative. 
Our system is built using \textbf{RoBERTa-base} with a regression head, trained on the official dataset, and evaluated using the official SemEval scoring script.

\subsection*{Task Overview}
SemEval Task 5 introduces the \textbf{AmbiStory} dataset, where each example contains:
\begin{itemize}
    \item Three precontext sentences
    \item One ambiguous sentence containing a homonym
    \item One optional ending sentence
    \item Two word senses with human plausibility ratings
    \item Mean and Standard Deviation of human annotations
\end{itemize}

The model must output a \textbf{continuous plausibility score} from 1 to 5, representing how plausible the meaning is in the narrative.

\subsection*{Installation}

\subsubsection*{1. Clone the repository}
\begin{verbatim}
git clone https://github.com/tanvinimbalkar/SemEval-2026-Task-5-Ambiguous-Story-Plausibility-Rating.git
cd Sem_Eval_Task-05
\end{verbatim}

\subsubsection*{2. Install dependencies}
\begin{verbatim}
pip install -r requirements.txt
\end{verbatim}

\subsection*{Training the Model}
Run the training script:
\begin{verbatim}
python src/train_model.py
\end{verbatim}

This will:
\begin{itemize}
    \item Load the RoBERTa tokenizer and base model
    \item Process \texttt{train.json} and \texttt{dev.json}
    \item Train for 3 epochs
    \item Save model config and tokenizer files to:
\end{itemize}

\begin{verbatim}
models/semeval_roberta/
\end{verbatim}

\subsection*{Generating Predictions (Dev Set)}

\begin{verbatim}
python src/generate_solution_from_dev.py
\end{verbatim}

This generates:
\begin{verbatim}
predictions_dev.jsonl
\end{verbatim}

Example line:
\begin{verbatim}
{"id": "0", "pred": 4.0}
\end{verbatim}

\subsection*{Formatting Predictions for Scoring}

SemEval requires:
\begin{verbatim}
{"id": "0", "prediction": 4.0}
\end{verbatim}

Convert predictions:
\begin{verbatim}
python src/fix_predictions.py
\end{verbatim}

Output:
\begin{verbatim}
predictions_dev_fixed.jsonl
\end{verbatim}

\subsection*{Evaluating Your Model (Official SemEval Script)}

\begin{verbatim}
python semeval26-05-scripts/scoring.py \
    semeval26-05-scripts/input/ref/solution.jsonl \
    predictions_dev_fixed.jsonl \
    scores.json
\end{verbatim}

Example Output:
\begin{verbatim}
{
  "spearman": 0.319,
  "within_std_accuracy": 0.474
}
\end{verbatim}

\noindent Spearman $\rightarrow$ ranking correlation between human scores and model predictions.\\
Within-Std Accuracy $\rightarrow$ prediction is within $\pm1$ standard deviation of the human mean.

\subsection*{Example Story (Illustration)}
\begin{verbatim}
Precontext: The detectives arrived at the abandoned train station...
Ambiguous Sentence: They followed the track.
Ending: They began to run along the abandoned railway line...
Homonym: track
Judged Meaning: "a pair of parallel rails..."
Average Human Rating: 3.6

Model output:
prediction = 4.0
\end{verbatim}

\subsection*{Requirements}
\begin{verbatim}
torch
transformers
datasets
numpy
scipy
pandas
tqdm
\end{verbatim}

\subsection*{Notes}
\begin{itemize}
    \item Model weights (\texttt{.bin}, \texttt{.safetensors}) are not included due to GitHub size limits.
    \item The repository includes the official scoring scripts for reproducibility.
    \item The notebook \texttt{semeval-task05.ipynb} contains the complete training pipeline used on Kaggle.
    \item Prediction file format strictly follows SemEval guidelines.
\end{itemize}
