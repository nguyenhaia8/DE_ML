"""Shared long-document embedding helpers for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_CHUNK_OVERLAP_TOKENS = 32
EMBEDDING_STRATEGY = "token_id_chunk_weighted_mean_pool_v1"


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_tokens: int
    overlap_tokens: int
    encoder_max_seq_length: int


@dataclass(frozen=True)
class TextChunk:
    token_ids: tuple[int, ...]
    token_count: int
    unique_token_count: int


def resolve_chunking_config(
    model,
    chunk_tokens: int | None = None,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> ChunkingConfig:
    """Resolve a safe content-token window for a SentenceTransformer model."""
    encoder_max_seq_length = int(model.max_seq_length)
    special_tokens = int(model.tokenizer.num_special_tokens_to_add(pair=False))
    if (
        special_tokens != 2
        or model.tokenizer.cls_token_id is None
        or model.tokenizer.sep_token_id is None
    ):
        raise ValueError(
            "Chunked token-ID encoding currently requires a BERT-style encoder "
            "with CLS and SEP tokens."
        )
    max_content_tokens = encoder_max_seq_length - special_tokens
    if max_content_tokens < 1:
        raise ValueError("The encoder has no room for content tokens.")

    resolved_chunk_tokens = max_content_tokens if chunk_tokens is None else int(chunk_tokens)
    if resolved_chunk_tokens < 1 or resolved_chunk_tokens > max_content_tokens:
        raise ValueError(
            f"chunk_tokens must be between 1 and {max_content_tokens} for this encoder."
        )

    resolved_overlap = int(overlap_tokens)
    if resolved_overlap < 0 or resolved_overlap >= resolved_chunk_tokens:
        raise ValueError("overlap_tokens must be non-negative and smaller than chunk_tokens.")

    return ChunkingConfig(
        chunk_tokens=resolved_chunk_tokens,
        overlap_tokens=resolved_overlap,
        encoder_max_seq_length=encoder_max_seq_length,
    )


def split_text_for_encoder(
    model,
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[TextChunk]:
    """Split all tokenizer tokens into overlapping windows without dropping a suffix."""
    tokenizer = model.tokenizer
    token_ids = tokenizer.encode(
        str(text),
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )
    if not token_ids:
        return [TextChunk(token_ids=(), token_count=0, unique_token_count=1)]

    stride = chunk_tokens - overlap_tokens
    chunks: list[TextChunk] = []
    covered_until = 0

    for start in range(0, len(token_ids), stride):
        end = min(start + chunk_tokens, len(token_ids))
        unique_token_count = max(0, end - covered_until)
        chunks.append(
            TextChunk(
                token_ids=tuple(token_ids[start:end]),
                token_count=end - start,
                unique_token_count=max(1, unique_token_count),
            )
        )
        covered_until = max(covered_until, end)
        if end == len(token_ids):
            break

    return chunks


def _encode_token_chunks(
    model,
    chunks: Sequence[TextChunk],
    *,
    batch_size: int,
    show_progress_bar: bool,
) -> np.ndarray:
    """Run exact token-ID windows through the complete SentenceTransformer stack."""
    import torch

    batch_starts = range(0, len(chunks), batch_size)
    if show_progress_bar:
        from tqdm.auto import tqdm

        batch_starts = tqdm(
            batch_starts,
            total=(len(chunks) + batch_size - 1) // batch_size,
            desc="Chunk batches",
        )

    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in batch_starts:
            batch = chunks[start : start + batch_size]
            prepared = []
            for chunk in batch:
                input_ids = [
                    model.tokenizer.cls_token_id,
                    *chunk.token_ids,
                    model.tokenizer.sep_token_id,
                ]
                prepared.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": [1] * len(input_ids),
                        "token_type_ids": [0] * len(input_ids),
                    }
                )
            features = model.tokenizer.pad(
                prepared,
                padding=True,
                return_tensors="pt",
            )
            features = {
                key: value.to(model.device)
                for key, value in features.items()
            }
            sentence_embeddings = model(features)["sentence_embedding"]
            embeddings.append(sentence_embeddings.detach().cpu().numpy())

    return np.vstack(embeddings)


def embed_texts_chunked(
    model,
    texts: Sequence[str],
    *,
    chunk_tokens: int | None = None,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> tuple[np.ndarray, dict[str, int | str]]:
    """Embed every token in each document, then pool its chunk embeddings."""
    config = resolve_chunking_config(model, chunk_tokens, overlap_tokens)
    document_chunks = [
        split_text_for_encoder(
            model,
            text,
            chunk_tokens=config.chunk_tokens,
            overlap_tokens=config.overlap_tokens,
        )
        for text in texts
    ]
    flat_chunks = [chunk for chunks in document_chunks for chunk in chunks]
    chunk_embeddings = _encode_token_chunks(
        model,
        flat_chunks,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    document_embeddings = []
    offset = 0
    for chunks in document_chunks:
        next_offset = offset + len(chunks)
        weights = np.asarray([chunk.unique_token_count for chunk in chunks], dtype=np.float32)
        pooled = np.average(chunk_embeddings[offset:next_offset], axis=0, weights=weights)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        document_embeddings.append(pooled.astype(np.float32, copy=False))
        offset = next_offset

    metadata: dict[str, int | str] = {
        "embedding_strategy": EMBEDDING_STRATEGY,
        "chunk_tokens": config.chunk_tokens,
        "chunk_overlap_tokens": config.overlap_tokens,
        "encoder_max_seq_length": config.encoder_max_seq_length,
    }
    return np.vstack(document_embeddings), metadata


def embed_text_with_model(
    model,
    text: str,
    *,
    chunk_tokens: int | None = None,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> np.ndarray:
    embeddings, _ = embed_texts_chunked(
        model,
        [text],
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
    )
    return embeddings


def embed_texts_with_model(
    model,
    texts: Sequence[str],
    *,
    chunk_tokens: int | None = None,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> np.ndarray:
    embeddings, _ = embed_texts_chunked(
        model,
        texts,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )
    return embeddings
