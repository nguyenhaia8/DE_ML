"""
IMPROVED MODEL
==============
Key upgrades over the baseline / over the original paper:

  1. Contextual sentence embeddings (Sentence-BERT) instead of averaged
     static GloVe vectors or TF-IDF+SVD — captures word order and meaning
     in context, not just bag-of-words frequency.
  2. XGBoost instead of plain Random Forest — usually a few points better
     on tabular/embedding features, and much faster to tune.
  3. SHAP explainability — the paper explicitly lists this as missing
     future work. We add TreeExplainer output so each prediction can be
     inspected ("which embedding dimensions pushed this trait up/down").

NOTE: the first run needs internet access to download the SBERT model
('all-MiniLM-L6-v2', ~80MB) from Hugging Face. After that it's cached
locally and works offline.

Usage:
    python train_improved.py --data essays.csv --out models/improved.joblib
"""
import argparse
import os
import joblib
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

from common import TRAIT_COLS, TRAIT_NAMES, load_essays, clean_text


def embed_texts(texts, model_name="all-MiniLM-L6-v2", batch_size=32):
    """Encode texts with Sentence-BERT. Returns (n_docs, embedding_dim) array."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings, model_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="essays.csv")
    parser.add_argument("--out", default="models/improved.joblib")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--skip-shap", action="store_true", help="Skip SHAP (faster)")
    args = parser.parse_args()

    print(f"Loading labeled essays from {args.data} ...")
    df = load_essays(args.data)
    texts = df["TEXT"].apply(clean_text).tolist()
    print(f"Loaded {len(texts)} labeled documents.")

    print(f"Encoding with Sentence-BERT ({args.embed_model}) ...")
    X, embed_model_name = embed_texts(texts, model_name=args.embed_model)
    print(f"Embedding matrix shape: {X.shape}")

    classifiers = {}
    results = {}
    shap_summaries = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\nTraining + 5-fold out-of-fold evaluation per trait:")
    print(f"{'Trait':<18}{'r (pred vs self-report)':>26}")
    print("-" * 44)

    for trait in TRAIT_COLS:
        y = df[trait].values

        clf_cv = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        oof_proba = cross_val_predict(clf_cv, X, y, cv=cv, method="predict_proba")[:, 1]
        r, _ = pearsonr(oof_proba, y)
        results[trait] = r
        print(f"{TRAIT_NAMES[trait]:<18}{r:>26.3f}")

        clf_final = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        clf_final.fit(X, y)
        classifiers[trait] = clf_final

        if not args.skip_shap:
            import shap

            explainer = shap.TreeExplainer(clf_final)
            shap_values = explainer.shap_values(X[:200])  # sample for speed
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            top_dims = np.argsort(mean_abs_shap)[::-1][:10]
            shap_summaries[trait] = {
                "top_dims": top_dims.tolist(),
                "top_dim_importance": mean_abs_shap[top_dims].tolist(),
            }

    avg_r = float(np.mean(list(results.values())))
    print("-" * 44)
    print(f"{'Average':<18}{avg_r:>26.3f}")

    bundle = {
        "embed_model_name": embed_model_name,
        "classifiers": classifiers,
        "trait_cols": TRAIT_COLS,
        "trait_names": TRAIT_NAMES,
        "model_type": "improved_sbert_xgboost",
        "cv_results": results,
        "shap_summaries": shap_summaries,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"\nSaved improved model bundle to {args.out}")
    if shap_summaries:
        print("SHAP summary (top influential embedding dims per trait) saved in the bundle.")


if __name__ == "__main__":
    main()
