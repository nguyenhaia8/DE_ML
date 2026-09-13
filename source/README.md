# CV Personality Prediction — Baseline + Improved Model

Re-implementation and improvement of:
Grunenberg et al. (2024), *"Machine learning in recruiting: predicting personality
from CVs and short text responses"*, Frontiers in Social Psychology.

## 1. Why two datasets?

`Resume.csv` (your CV data) has **no Big Five labels** — it only has
`ID, Resume_str, Resume_html, Category`. You can't train a personality model
directly on it. Same problem the original paper solved by training on a
*different* proprietary recruiting dataset.

We do the same thing with public data:

| Dataset | Role | Has personality labels? |
|---|---|---|
| `essays.csv` (Pennebaker & King corpus, 2,467 essays) | **Train** the personality model | Yes — cEXT/cNEU/cAGR/cCON/cOPN |
| `Resume.csv` (your file) | **Apply** the trained model + validate | No — but has `Category` (job field), used the same way the paper used vocational interests |

Get `essays.csv`:
```bash
curl -L -o essays.csv https://raw.githubusercontent.com/SenticNet/personality-detection/master/essays.csv
```

## 2. Install

```bash
pip install -r requirements.txt
```
(The improved model needs internet on first run to download the Sentence-BERT
weights from Hugging Face — cached locally afterward.)

## 3. Train

```bash
# Baseline: TF-IDF + SVD + Random Forest (closest to the paper's original method)
python train_baseline.py --data essays.csv --out models/baseline.joblib

# Improved: Sentence-BERT + XGBoost + SHAP
python train_improved.py --data essays.csv --out models/improved.joblib
```

Both scripts print 5-fold out-of-fold correlation (r) between predicted and
self-reported trait per Big Five dimension — the same metric the paper
reports in Figure 2. Compare the two printouts to show your improvement.

## 4. Score your CVs + validate against Category

```bash
python predict_and_compare.py --model models/baseline.joblib  --resumes Resume.csv --out predictions_baseline.csv
python predict_and_compare.py --model models/improved.joblib  --resumes Resume.csv --out predictions_improved.csv
```

This prints mean predicted trait score per job `Category` — check whether
the pattern makes intuitive sense (e.g. Sales/PR scoring higher on
Extraversion or Openness than Accounting). This is your criterion-validity
check, standing in for the paper's vocational-interest table.

Use `--sample 200` while iterating — full-dataset scoring with the improved
(SBERT) model is slower than the baseline.

## 5. Integrate into the demo app

```bash
streamlit run app.py
```

- Sidebar: switch between baseline / improved model, see each one's
  validation accuracy.
- Upload a CV PDF or paste text → radar chart of predicted Big Five.

## 6. Integrating the model into your own backend (not Streamlit)

Minimal inference snippet — this is all `app.py` does under the hood:

```python
import joblib
from common import clean_text, TRAIT_NAMES

bundle = joblib.load("models/improved.joblib")   # or baseline.joblib

def predict_personality(raw_text: str) -> dict:
    text = clean_text(raw_text)

    if bundle["model_type"] == "baseline_tfidf_svd_rf":
        X = bundle["svd"].transform(bundle["vectorizer"].transform([text]))
    else:  # improved_sbert_xgboost
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(bundle["embed_model_name"])
        X = embedder.encode([text])

    return {
        TRAIT_NAMES[trait]: float(clf.predict_proba(X)[0, 1])
        for trait, clf in bundle["classifiers"].items()
    }
```

Wrap this in a FastAPI route (`POST /predict`, body: `{"text": "..."}`) if you
want a REST API instead of / alongside the Streamlit UI — same `bundle`,
same function, just called from a route handler instead of a button click.

## 7. What's actually improved vs. the original paper

| | Paper (2024) | This project |
|---|---|---|
| Text features | GloVe averaged + TF-IDF/SVD + LIWC + LDA (694 features) | TF-IDF/SVD (baseline) **or** contextual Sentence-BERT (improved) |
| Model | Random Forest / Lasso | Random Forest (baseline) **or** XGBoost (improved) |
| Explainability | None (listed as future work) | SHAP values on the improved model |
| Deployment | None (research only) | Streamlit app + REST-ready inference function |
| Criterion validity | Self-reported vocational interests | Job `Category` field on real resumes |

## Files

```
common.py                shared constants + data loading
train_baseline.py        TF-IDF + SVD + Random Forest, per trait
train_improved.py        Sentence-BERT + XGBoost + SHAP, per trait
predict_and_compare.py   score Resume.csv, validate against Category
extract_cv_text.py       PDF -> text (for real CV uploads)
app.py                   Streamlit demo
requirements.txt
```
