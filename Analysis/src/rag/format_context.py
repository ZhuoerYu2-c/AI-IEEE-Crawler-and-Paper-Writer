"""
格式化检索上下文，并为证据块分配稳定引用标签。
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def format_context(
    chunks: List[Dict],
    max_chars: int = 9000,
    ref_prefix: str = "Ref",
) -> Tuple[str, List[Dict]]:
    """
    将检索结果格式化为上下文字符串，并返回对应的引用记录。

    返回：
    - context_text: 供 LLM 阅读的上下文
    - reference_records: 与上下文中标签一一对应的元数据表
    """
    parts: List[str] = []
    records: List[Dict] = []
    total = 0

    for i, c in enumerate(chunks, 1):
        ref_id = f"{ref_prefix}{i}"
        block = (
            f"[{ref_id}]\n"
            f"source={c.get('source')}\n"
            f"title={c.get('title')}\n"
            f"chunk_id={c.get('chunk_id')}\n"
            f"span=({c.get('start_char')},{c.get('end_char')})\n"
            f"score={c.get('score', 0.0):.4f}\n"
            f"text:\n{c.get('text')}\n"
        )
        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)
        records.append(
            {
                "ref_id": ref_id,
                "source": c.get("source"),
                "title": c.get("title"),
                "chunk_id": c.get("chunk_id"),
                "start_char": c.get("start_char"),
                "end_char": c.get("end_char"),
                "score": float(c.get("score", 0.0)),
                "text": c.get("text", ""),
            }
        )

    return "\n---\n".join(parts), records
