"""
BASELINE MODEL
==============
Mirrors the original paper's approach as closely as possible without needing
a GloVe download: word/n-gram frequencies (TF-IDF) reduced via SVD, fed into
a Random Forest — one classifier per Big Five trait.

Usage:
    python train_baseline.py --data essays.csv --out models/baseline.joblib

Output:
    - A trained model bundle (vectorizer + SVD + 5 classifiers) saved with joblib
    - Printed per-trait validation performance (5-fold out-of-fold correlation
      between predicted probability and self-reported label — same evaluation
      style the paper uses: r between predicted and self-report score)
"""
import argparse
import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from common import TRAIT_COLS, TRAIT_NAMES, load_essays, clean_text


def build_features(texts, max_features=5000, n_components=100, vectorizer=None, svd=None):
    """Fit (or reuse) TF-IDF + SVD feature pipeline."""
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    if svd is None:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        features = svd.fit_transform(tfidf)
    else:
        features = svd.transform(tfidf)

    return features, vectorizer, svd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="essays.csv")
    parser.add_argument("--out", default="models/baseline.joblib")
    parser.add_argument("--n-components", type=int, default=100)
    args = parser.parse_args()

    print(f"Loading labeled essays from {args.data} ...")
    df = load_essays(args.data)
    texts = df["TEXT"].apply(clean_text).tolist()
    print(f"Loaded {len(texts)} labeled documents.")

    print("Building TF-IDF + SVD features ...")
    X, vectorizer, svd = build_features(texts, n_components=args.n_components)

    classifiers = {}
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nTraining + 5-fold out-of-fold evaluation per trait:")
    print(f"{'Trait':<18}{'r (pred vs self-report)':>26}")
    print("-" * 44)

    for trait in TRAIT_COLS:
        y = df[trait].values

        # Out-of-fold predicted probabilities (unbiased estimate, like the
        # paper's nested cross-validation outer loop).
        clf_cv = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        oof_proba = cross_val_predict(clf_cv, X, y, cv=cv, method="predict_proba")[:, 1]
        r, _ = pearsonr(oof_proba, y)
        results[trait] = r
        print(f"{TRAIT_NAMES[trait]:<18}{r:>26.3f}")

        # Refit on all data for the deployed model.
        clf_final = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf_final.fit(X, y)
        classifiers[trait] = clf_final

    avg_r = float(np.mean(list(results.values())))
    print("-" * 44)
    print(f"{'Average':<18}{avg_r:>26.3f}")

    bundle = {
        "vectorizer": vectorizer,
        "svd": svd,
        "classifiers": classifiers,
        "trait_cols": TRAIT_COLS,
        "trait_names": TRAIT_NAMES,
        "model_type": "baseline_tfidf_svd_rf",
        "cv_results": results,
    }

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"\nSaved baseline model bundle to {args.out}")


if __name__ == "__main__":
    main()
