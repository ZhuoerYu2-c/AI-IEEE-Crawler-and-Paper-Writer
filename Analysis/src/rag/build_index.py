from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np

from .embed import QwenEmbedder


def build_faiss_index(
    chunks: List[Dict],
    index_dir: Path,
    emb_api_key: str,
    emb_model: str = "text-embedding-v4",
    emb_dimensions: int = 1024,
    emb_batch_size: int = 10,
) -> None:
    """构建 FAISS 向量索引"""
    index_dir.mkdir(parents=True, exist_ok=True)

    embedder = QwenEmbedder(
        api_key=emb_api_key,
        model=emb_model,
        dimensions=emb_dimensions,
        batch_size=emb_batch_size,
    )

    texts = [c["text"] for c in chunks]
    print(f"📊 正在生成 {len(texts)} 个文本块的向量...")
    vecs = embedder.encode(texts)

    if not vecs or len(vecs) == 0:
        raise RuntimeError("❌ 向量化结果为空，无法构建索引。")

    # 转换为 numpy 数组
    vecs = np.array(vecs, dtype="float32")
    
    # 确保向量维度正确
    if vecs.ndim != 2:
        raise RuntimeError(f"❌ 向量维度错误: {vecs.shape}")

    d = vecs.shape[1]

    # 使用内积索引（余弦相似度需要先归一化，这里直接用）
    index = faiss.IndexFlatIP(d)
    index.add(vecs)

    # 保存索引
    faiss.write_index(index, str(index_dir / "faiss.index"))

    # 保存元数据
    meta_path = index_dir / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"✅ 索引已保存: {index_dir / 'faiss.index'}")
    print(f"✅ 元数据已保存: {meta_path}")
