"""
RAG 检索模块
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from .embed import QwenEmbedder


def _load_meta(meta_jsonl: Path) -> List[Dict]:
    """加载元数据"""
    items: List[Dict] = []
    with meta_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def retrieve(
    query: str,
    index_dir: Path,
    emb_api_key: str,
    emb_model: str = "text-embedding-v4",
    emb_dimensions: int = 1024,
    emb_batch_size: int = 10,
    top_k: int = 8,
) -> List[Dict]:
    """
    从向量索引中检索相关文本块
    """
    index_path = index_dir / "faiss.index"
    meta_path = index_dir / "meta.jsonl"

    if not index_path.exists() or not meta_path.exists():
        return []

    # 加载索引
    index = faiss.read_index(str(index_path))
    meta = _load_meta(meta_path)

    # 查询向量
    embedder = QwenEmbedder(
        api_key=emb_api_key,
        model=emb_model,
        dimensions=emb_dimensions,
        batch_size=emb_batch_size,
    )
    
    qv = embedder.encode([query])
    qv = np.array(qv, dtype="float32")

    # 检索
    scores, ids = index.search(qv, top_k)
    
    results: List[Dict] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(meta[idx])
        item["score"] = float(score)
        results.append(item)
    
    return results
