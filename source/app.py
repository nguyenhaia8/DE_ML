"""
Streamlit demo app.

Run with:
    streamlit run app.py

Lets a user upload a CV (PDF) or paste text, choose which trained model to
use (baseline vs improved), and see predicted Big Five scores as a radar
chart plus (for the improved model) a short explanation of which signal
dimensions mattered most.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from common import TRAIT_COLS, TRAIT_NAMES, clean_text
from embedding_utils import (
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    EMBEDDING_STRATEGY,
    embed_text_with_model,
)
from extract_cv_text import extract_text_from_pdf

st.set_page_config(page_title="CV Personality Predictor", layout="centered")
st.title("CV → Predicted Big Five Personality")
st.caption(
    "Improved re-implementation of Grunenberg et al. (2024), "
    "'Machine learning in recruiting: predicting personality from CVs and short text responses'."
)

MODEL_DIR = "models"


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


@st.cache_resource
def get_sbert_model(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def predict_with_bundle(bundle, text):
    if bundle["model_type"] == "baseline_tfidf_svd_rf":
        X = bundle["svd"].transform(bundle["vectorizer"].transform([text]))
    elif bundle["model_type"] == "improved_sbert_xgboost":
        if bundle.get("embedding_strategy") != EMBEDDING_STRATEGY:
            raise ValueError(
                "Improved model was trained without full-document chunk pooling. Retrain it before serving predictions."
            )
        model = get_sbert_model(bundle["embed_model_name"])
        X = embed_text_with_model(
            model,
            text,
            chunk_tokens=bundle.get("chunk_tokens"),
            overlap_tokens=bundle.get(
                "chunk_overlap_tokens", DEFAULT_CHUNK_OVERLAP_TOKENS
            ),
        )
    else:
        raise ValueError(f"Unknown model_type: {bundle['model_type']}")

    scores = {}
    for trait in bundle["trait_cols"]:
        proba = bundle["classifiers"][trait].predict_proba(X)[0, 1]
        scores[TRAIT_NAMES[trait]] = float(proba)
    return scores


def radar_chart(scores: dict, title: str):
    categories = list(scores.keys())
    values = list(scores.values())
    values += values[:1]
    categories += categories[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values, theta=categories, fill="toself", name=title))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=title,
    )
    return fig


# --- Sidebar: model selection ---
available_models = {}
if os.path.exists(os.path.join(MODEL_DIR, "baseline.joblib")):
    available_models["Baseline (TF-IDF + SVD + Random Forest)"] = os.path.join(MODEL_DIR, "baseline.joblib")
if os.path.exists(os.path.join(MODEL_DIR, "improved.joblib")):
    available_models["Improved (Sentence-BERT + XGBoost)"] = os.path.join(MODEL_DIR, "improved.joblib")

if not available_models:
    st.error(
        "No trained models found in ./models. Run `python train_baseline.py` "
        "and/or `python train_improved.py` first."
    )
    st.stop()

model_choice = st.sidebar.selectbox("Model", list(available_models.keys()))
bundle = load_bundle(available_models[model_choice])
if bundle.get("cv_results"):
    st.sidebar.markdown("**Validation accuracy (r, out-of-fold):**")
    for trait, r in bundle["cv_results"].items():
        st.sidebar.write(f"{TRAIT_NAMES[trait]}: {r:.3f}")

# --- Main: input ---
tab1, tab2 = st.tabs(["Upload PDF", "Paste text"])
cv_text = None

with tab1:
    uploaded = st.file_uploader("Upload a CV (PDF)", type=["pdf"])
    if uploaded is not None:
        with st.spinner("Extracting text..."):
            cv_text = extract_text_from_pdf(uploaded)
        st.text_area("Extracted text (preview)", cv_text[:1000] + "...", height=150)

with tab2:
    pasted = st.text_area("Or paste CV / bio text here", height=200)
    if pasted.strip():
        cv_text = pasted

if cv_text and st.button("Predict personality"):
    with st.spinner("Scoring..."):
        text = clean_text(cv_text)
        if len(text.split()) < 20:
            st.warning("Text looks very short — predictions may be unreliable (paper excluded CVs under 50 words).")
        scores = predict_with_bundle(bundle, text)

    st.plotly_chart(radar_chart(scores, model_choice), use_container_width=True)

    st.subheader("Scores")
    score_df = pd.DataFrame({"Trait": list(scores.keys()), "Predicted score (0-1)": list(scores.values())})
    st.dataframe(score_df.set_index("Trait").style.format("{:.2f}"))

    st.caption(
        "Scores are model-estimated probabilities (0 = low trait, 1 = high trait), "
        "not a clinical or definitive personality assessment. "
        "Use for exploration / research only, per the original paper's caution "
        "against using such models as final hiring decision tools."
    )
