from bert_score import score
from rouge_score import rouge_scorer
import torch
import numpy as np

def compute_metrics(eval_df,
                    pred_col="generated",
                    ref_col="Norwegian target response"):

    eval_df = eval_df.copy()

    eval_df[pred_col] = eval_df[pred_col].fillna("").astype(str)
    eval_df[ref_col] = eval_df[ref_col].fillna("").astype(str)

    eval_df = eval_df[
        (eval_df[pred_col].str.strip() != "") &
        (eval_df[ref_col].str.strip() != "")
    ].copy()

    preds = eval_df[pred_col].tolist()
    refs  = eval_df[ref_col].tolist()

    rouge_scorer_l = rouge_scorer.RougeScorer(
        ["rougeL"], use_stemmer=False
    )

# ROUGE-l 
    rouge_l_f1_scores = []
    for p, r in zip(preds, refs):
        scores = rouge_scorer_l.score(r, p)
        rouge_l_f1_scores.append(scores["rougeL"].fmeasure)

    rouge_l_f1 = float(np.mean(rouge_l_f1_scores))

# BERTScore
    _, _, F1 = score(preds, refs, lang="no")
    bertscore_f1 = F1.mean().item()

    return {
        "ROUGE-L (F1)": rouge_l_f1,
        "BERTScore (F1)": bertscore_f1
    }


# Function for using the classifier to classify responses
def classify_responses(df, response_col, model, tokenizer):

    texts = df[response_col]

    predictions = []

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model(**inputs)

        pred_class = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(pred_class)

    df["predicted_label"] = predictions

    return df