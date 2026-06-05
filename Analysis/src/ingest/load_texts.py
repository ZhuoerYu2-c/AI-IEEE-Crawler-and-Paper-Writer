"""
文本加载模块
"""
from pathlib import Path
from typing import List, Tuple


def load_raw_texts(raw_dir: Path) -> List[Tuple[Path, str]]:
    """加载原始文本文件"""
    if not raw_dir.exists():
        return []
    
    items: List[Tuple[Path, str]] = []
    
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".md", ".txt"}:
            try:
                text = p.read_text(encoding="utf-8")
                if text.strip():
                    items.append((p, text))
            except Exception as e:
                print(f"⚠️ 读取失败 {p}: {e}")
    
    return sorted(items, key=lambda x: x[0].name)
