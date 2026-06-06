"""
Text embeddings for researchers (abstracts) and grants (guidelines).
Prefers SPECTER2; falls back to MiniLM if unavailable.
Caches vectors under data/cache/ to avoid re-encoding on every run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SPECTER2_MODEL = "allenai/specter2_base"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _load_encoder():
    from sentence_transformers import SentenceTransformer

    for name in (SPECTER2_MODEL, "allenai/specter2", MINILM_MODEL):
        try:
            model = SentenceTransformer(name)
            print(f"[Embeddings] Using model: {name} (dim={model.get_sentence_embedding_dimension()})")
            return model, name
        except Exception as e:
            print(f"[Embeddings] Could not load {name}: {e}")
    raise RuntimeError("No embedding model available. pip install sentence-transformers")


def researcher_text(r: dict) -> str:
    parts = r.get("recent_abstracts") or []
    if isinstance(parts, list) and parts:
        return " [SEP] ".join(str(p) for p in parts[:5])
    return r.get("name", "") or "unknown researcher"


def grant_text(g: dict) -> str:
    title = g.get("title", "")
    body = g.get("guidelines_text") or g.get("guidelines", "") or ""
    return f"{title} {body}".strip() or title or "unknown grant"


def _cache_key(model_name: str, texts: List[str]) -> Path:
    h = hashlib.sha256((model_name + "|" + str(len(texts)) + "|" + texts[0][:200] if texts else "").encode())
    digest = h.hexdigest()[:16]
    return CACHE_DIR / f"emb_{digest}_{len(texts)}.npz"


def encode_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    model, model_name = _load_encoder()
    cache_path = _cache_key(model_name, texts)
    if cache_path.exists():
        arr = np.load(cache_path)["embeddings"]
        if arr.shape[0] == len(texts):
            print(f"[Embeddings] Loaded cache {cache_path.name} shape={arr.shape}")
            return arr

    print(f"[Embeddings] Encoding {len(texts)} texts...")
    arr = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 50,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    np.savez_compressed(cache_path, embeddings=arr)
    meta = {"model": model_name, "dim": int(arr.shape[1]), "count": len(texts)}
    with open(cache_path.with_suffix(".json"), "w") as f:
        json.dump(meta, f)
    return arr


def aggregate_researcher_embedding(r: dict, model, dim: int) -> np.ndarray:
    """Mean-pool paper-level embeddings when multiple abstracts exist."""
    abstracts = r.get("recent_abstracts") or []
    if not abstracts:
        text = researcher_text(r)
        return model.encode([text], convert_to_numpy=True)[0]
    embs = model.encode(abstracts[:30], convert_to_numpy=True, show_progress_bar=False)
    return np.mean(embs, axis=0)


def build_node_embeddings(
    researchers: List[dict],
    grants: List[dict],
    use_aggregate_abstracts: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Returns (researcher_emb [N_r, D], grant_emb [N_g, D], dim)."""
    from sentence_transformers import SentenceTransformer

    model, model_name = _load_encoder()
    dim = model.get_sentence_embedding_dimension()

    cache_r = CACHE_DIR / f"r_agg_{len(researchers)}_{model_name.split('/')[-1]}.npz"
    cache_g = CACHE_DIR / f"g_{len(grants)}_{model_name.split('/')[-1]}.npz"

    if use_aggregate_abstracts and cache_r.exists():
        r_arr = np.load(cache_r)["embeddings"]
    else:
        if use_aggregate_abstracts:
            print("[Embeddings] Aggregating paper abstracts per researcher...")
            r_arr = np.vstack(
                [aggregate_researcher_embedding(r, model, dim) for r in researchers]
            )
            np.savez_compressed(cache_r, embeddings=r_arr)
        else:
            r_arr = encode_texts([researcher_text(r) for r in researchers])

    if cache_g.exists():
        g_arr = np.load(cache_g)["embeddings"]
        if g_arr.shape[0] != len(grants):
            g_arr = encode_texts([grant_text(g) for g in grants])
    else:
        g_arr = encode_texts([grant_text(g) for g in grants])
        np.savez_compressed(cache_g, embeddings=g_arr)

    assert g_arr.shape[1] == r_arr.shape[1]
    with_abstracts = sum(1 for r in researchers if r.get("recent_abstracts"))
    print(f"[Embeddings] Researchers with abstracts: {with_abstracts}/{len(researchers)}")
    return (
        torch.tensor(r_arr, dtype=torch.float32),
        torch.tensor(g_arr, dtype=torch.float32),
        int(r_arr.shape[1]),
    )
