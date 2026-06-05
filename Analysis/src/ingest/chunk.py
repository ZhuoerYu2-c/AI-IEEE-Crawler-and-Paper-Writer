"""
文本分块模块
"""
import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Chunk:
    chunk_id: str
    source: str
    title: str
    start_char: int
    end_char: int
    text: str


def _path_safe(s: str) -> str:
    """路径安全化"""
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", s)[-80:]


def split_by_headings(text: str) -> List[Tuple[str, int, int, str]]:
    """按标题分割文本"""
    heading_pat = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pat.finditer(text))
    
    if not matches:
        return [("Untitled", 0, len(text), text)]

    sections = []
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((title, start, end, text[start:end]))
    return sections


def sliding_window(text: str, max_chars: int, overlap: int, min_chars: int) -> List[Tuple[int, int, str]]:
    """滑动窗口分块"""
    out = []
    n = len(text)
    i = 0
    while i < n:
        j = min(i + max_chars, n)
        piece = text[i:j].strip()
        if len(piece) >= min_chars:
            out.append((i, j, piece))
        if j == n:
            break
        i = max(0, j - overlap)
    return out


def make_chunks(
    source: str,
    text: str,
    max_chars: int,
    overlap: int,
    min_chars: int,
    split_on_headings: bool,
) -> List[Chunk]:
    """生成文本块"""
    chunks: List[Chunk] = []
    idx = 0

    if split_on_headings:
        sections = split_by_headings(text)
        for title, s, _, sec_text in sections:
            for a, b, piece in sliding_window(sec_text, max_chars, overlap, min_chars):
                chunks.append(
                    Chunk(
                        chunk_id=f"{_path_safe(source)}_{idx}",
                        source=source,
                        title=title or "Untitled",
                        start_char=s + a,
                        end_char=s + b,
                        text=piece,
                    )
                )
                idx += 1
    else:
        for a, b, piece in sliding_window(text, max_chars, overlap, min_chars):
            chunks.append(
                Chunk(
                    chunk_id=f"{_path_safe(source)}_{idx}",
                    source=source,
                    title="Untitled",
                    start_char=a,
                    end_char=b,
                    text=piece,
                )
            )
            idx += 1

    return chunks
