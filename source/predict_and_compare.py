"""
Apply a trained model to Resume.csv (which has NO personality labels) and
check whether predicted traits line up sensibly with the `Category` column
(job field: HR, IT, SALES, DESIGNER, ...).

This plays the same role as the original paper's "vocational interests"
criterion-validity check (RQ2): we don't have ground-truth personality for
these CVs, so instead we test whether predictions correlate believably with
an independent external signal (here, job category) — e.g. does the model
predict higher Extraversion for Sales/PR resumes than for Engineering ones?

Usage:
    python predict_and_compare.py --model models/baseline.joblib --resumes Resume.csv --out predictions_baseline.csv
    python predict_and_compare.py --model models/improved.joblib --resumes Resume.csv --out predictions_improved.csv
"""
import argparse
import joblib
import numpy as np
import pandas as pd

from common import TRAIT_COLS, TRAIT_NAMES, load_resumes, clean_text


def predict_baseline(bundle, texts):
    X = bundle["svd"].transform(bundle["vectorizer"].transform(texts))
    preds = {}
    for trait in bundle["trait_cols"]:
        preds[trait] = bundle["classifiers"][trait].predict_proba(X)[:, 1]
    return preds


def predict_improved(bundle, texts):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(bundle["embed_model_name"])
    X = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    preds = {}
    for trait in bundle["trait_cols"]:
        preds[trait] = bundle["classifiers"][trait].predict_proba(X)[:, 1]
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a trained .joblib bundle")
    parser.add_argument("--resumes", default="Resume.csv")
    parser.add_argument("--out", default="predictions.csv")
    parser.add_argument("--sample", type=int, default=None, help="Optional: only score N resumes (for a quick test)")
    args = parser.parse_args()

    print(f"Loading model bundle from {args.model} ...")
    bundle = joblib.load(args.model)

    print(f"Loading resumes from {args.resumes} ...")
    df = load_resumes(args.resumes)
    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
    texts = df["Resume_str"].apply(clean_text).tolist()
    print(f"Scoring {len(texts)} resumes with model_type={bundle['model_type']} ...")

    if bundle["model_type"] == "baseline_tfidf_svd_rf":
        preds = predict_baseline(bundle, texts)
    elif bundle["model_type"] == "improved_sbert_xgboost":
        preds = predict_improved(bundle, texts)
    else:
        raise ValueError(f"Unknown model_type: {bundle['model_type']}")

    out_df = df[["ID", "Category"]].copy()
    for trait in TRAIT_COLS:
        out_df[TRAIT_NAMES[trait]] = preds[trait]
    out_df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

    # Criterion validity: mean predicted trait score per job Category,
    # analogous to Table 2 in the paper (personality x vocational interest).
    print("\nMean predicted trait score by job Category (criterion-validity check):")
    trait_display_cols = [TRAIT_NAMES[t] for t in TRAIT_COLS]
    summary = out_df.groupby("Category")[trait_display_cols].mean().round(3)
    print(summary.to_string())

    # Flag the category with the highest/lowest mean for each trait —
    # a quick sanity check (e.g. does Sales score highest on Extraversion?).
    print("\nHighest / lowest category per trait:")
    for col in trait_display_cols:
        top_cat = summary[col].idxmax()
        bottom_cat = summary[col].idxmin()
        print(f"  {col:<18} highest: {top_cat:<20} lowest: {bottom_cat}")


if __name__ == "__main__":
    main()
