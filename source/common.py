"""
Shared constants and helpers for the personality-prediction project.
Used by train_baseline.py, train_improved.py, predict_and_compare.py, and app.py.
"""
import re
import pandas as pd

# Trait column names as they appear in the public "Essays" Big-Five dataset
# (Pennebaker & King corpus, mirrored at SenticNet/personality-detection).
TRAIT_COLS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]

TRAIT_NAMES = {
    "cEXT": "Extraversion",
    "cNEU": "Neuroticism",
    "cAGR": "Agreeableness",
    "cCON": "Conscientiousness",
    "cOPN": "Openness",
}


def load_essays(path: str = "essays.csv") -> pd.DataFrame:
    """Load the labeled training corpus (essays with self-reported Big Five)."""
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]
    for c in TRAIT_COLS:
        df[c] = df[c].str.strip().map({"y": 1, "n": 0})
    df = df.dropna(subset=TRAIT_COLS + ["TEXT"]).reset_index(drop=True)
    for c in TRAIT_COLS:
        df[c] = df[c].astype(int)
    return df


def load_resumes(path: str = "Resume.csv") -> pd.DataFrame:
    """Load the unlabeled CV corpus we want to score (no Big Five ground truth)."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["Resume_str"]).reset_index(drop=True)
    return df


def clean_text(text: str) -> str:
    """Light normalization: collapse whitespace/newlines. Keep punctuation —
    TF-IDF and sentence embedders both benefit from sentence boundaries."""
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text
