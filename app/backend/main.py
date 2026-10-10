from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = PROJECT_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from embedding_utils import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP_TOKENS,
    EMBEDDING_STRATEGY,
    embed_text_with_model,
)

TRAIT_COLS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
TRAIT_NAMES = {
    "cEXT": "Extraversion",
    "cNEU": "Neuroticism",
    "cAGR": "Agreeableness",
    "cCON": "Conscientiousness",
    "cOPN": "Openness",
}

MODEL_DIR = PROJECT_ROOT / "source" / "models"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline.joblib"
IMPROVED_MODEL_PATH = MODEL_DIR / "improved.joblib"


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_id: Literal["baseline", "improved"] = "baseline"


class TraitScore(BaseModel):
    id: str
    label: str
    score: float


class ModelInfo(BaseModel):
    id: str
    name: str
    path: str


class PredictionResponse(BaseModel):
    model: ModelInfo
    source: Literal["text", "pdf"]
    word_count: int
    truncated: bool
    warnings: list[str]
    scores: list[TraitScore]


class PdfExtractionResponse(BaseModel):
    filename: str | None
    text: str
    word_count: int


@dataclass(frozen=True)
class ModelConfig:
    id: Literal["baseline", "improved"]
    name: str
    path: Path


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "baseline": ModelConfig(
        id="baseline",
        name="Baseline TF-IDF + SVD + Random Forest",
        path=BASELINE_MODEL_PATH,
    ),
    "improved": ModelConfig(
        id="improved",
        name="Improved Sentence-BERT + XGBoost",
        path=IMPROVED_MODEL_PATH,
    ),
}

app = FastAPI(title="CV Personality Prediction API", version="0.1.0")
allowed_origins = os.environ.get(
    "CV_PERSONALITY_ALLOWED_ORIGINS",
    "http://127.0.0.1:5173,http://localhost:5173",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins.split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_bundle_cache: dict[str, dict] = {}
_embedder_cache: dict[str, object] = {}


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "CV Personality Prediction API", "docs": "/docs"}


@app.get("/api/v1/health/ready")
def ready() -> dict[str, object]:
    return {
        "ok": True,
        "models": {
            model_id: config.path.exists()
            for model_id, config in MODEL_CONFIGS.items()
        },
    }


@app.get("/api/v1/models")
def models() -> list[dict[str, object]]:
    return [
        {
            "id": config.id,
            "name": config.name,
            "available": config.path.exists(),
            "path": str(config.path),
        }
        for config in MODEL_CONFIGS.values()
    ]


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_from_text(request: PredictionRequest) -> PredictionResponse:
    text = clean_text(request.text)
    return predict(text=text, model_id=request.model_id, source="text")


@app.post("/api/v1/extract/pdf", response_model=PdfExtractionResponse)
async def extract_from_pdf(file: UploadFile = File(...)) -> PdfExtractionResponse:
    validate_pdf_upload(file)
    raw_bytes = await file.read()
    text = clean_text(extract_pdf_text(raw_bytes))
    return PdfExtractionResponse(
        filename=file.filename,
        text=text,
        word_count=count_words(text),
    )


@app.post("/api/v1/predict/pdf", response_model=PredictionResponse)
async def predict_from_pdf(
    file: UploadFile = File(...),
    model_id: Literal["baseline", "improved"] = Form("baseline"),
) -> PredictionResponse:
    validate_pdf_upload(file)
    raw_bytes = await file.read()
    text = extract_pdf_text(raw_bytes)
    return predict(text=clean_text(text), model_id=model_id, source="pdf")


def predict(
    text: str,
    model_id: Literal["baseline", "improved"],
    source: Literal["text", "pdf"],
) -> PredictionResponse:
    word_count = count_words(text)
    if word_count < 20:
        raise HTTPException(status_code=400, detail="CV text must contain at least 20 words.")

    config = get_model_config(model_id)
    bundle = load_bundle(config)
    features = featurize(bundle, text)
    scores = [
        TraitScore(
            id=trait,
            label=TRAIT_NAMES[trait],
            score=float(bundle["classifiers"][trait].predict_proba(features)[0, 1]),
        )
        for trait in bundle["trait_cols"]
    ]

    return PredictionResponse(
        model=ModelInfo(id=config.id, name=config.name, path=str(config.path)),
        source=source,
        word_count=word_count,
        truncated=False,
        warnings=[],
        scores=scores,
    )


def get_model_config(model_id: str) -> ModelConfig:
    config = MODEL_CONFIGS.get(model_id)
    if config is None:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {model_id}")
    if not config.path.exists():
        raise HTTPException(status_code=503, detail=f"Model file not found: {config.path}")
    return config


def load_bundle(config: ModelConfig) -> dict:
    if config.id not in _bundle_cache:
        try:
            _bundle_cache[config.id] = joblib.load(config.path)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not load {config.id} model: {exc}") from exc
    return _bundle_cache[config.id]


def featurize(bundle: dict, text: str):
    model_type = bundle.get("model_type")
    if model_type == "baseline_tfidf_svd_rf":
        return bundle["svd"].transform(bundle["vectorizer"].transform([text]))

    if model_type == "improved_sbert_xgboost":
        if bundle.get("embedding_strategy") != EMBEDDING_STRATEGY:
            raise HTTPException(
                status_code=503,
                detail="Improved model was trained without full-document chunk pooling. Retrain it before serving predictions.",
            )
        model_name = bundle["embed_model_name"]
        if model_name not in _embedder_cache:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise HTTPException(
                    status_code=503,
                    detail="Improved model requires sentence-transformers. Install the optional improved-model dependencies first.",
                ) from exc
            _embedder_cache[model_name] = SentenceTransformer(model_name)
        return embed_text_with_model(
            _embedder_cache[model_name],
            text,
            chunk_tokens=bundle.get("chunk_tokens"),
            overlap_tokens=bundle.get(
                "chunk_overlap_tokens", DEFAULT_CHUNK_OVERLAP_TOKENS
            ),
        )

    raise HTTPException(status_code=503, detail=f"Unsupported model_type: {model_type}")


def extract_pdf_text(raw_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="PDF extraction requires pdfplumber.") from exc

    text_parts = []
    try:
        with pdfplumber.open(BytesIO(raw_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not extract text from PDF: {exc}") from exc

    extracted = "\n".join(text_parts)
    if not clean_text(extracted):
        raise HTTPException(status_code=400, detail="No readable text found in PDF.")
    return extracted


def validate_pdf_upload(file: UploadFile) -> None:
    filename = file.filename or ""
    if file.content_type not in {None, "", "application/pdf"} and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")


def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    return len([part for part in text.split() if part])
