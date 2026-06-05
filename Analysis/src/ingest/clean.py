"""
文本清洗模块
"""
import re


def basic_clean(text: str) -> str:
    """基础清洗"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 去除多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除页码
    text = re.sub(r"^\s*Page\s+\d+\s*(of\s+\d+)?\s*$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"^\s*\d+\s*/\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # 去除行尾空格
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def truncate_references(text: str) -> str:
    """截断参考文献部分"""
    markers = ["REFERENCES", "References", "参考文献", "BIBLIOGRAPHY"]
    idxs = [text.find(m) for m in markers if text.find(m) != -1]
    if not idxs:
        return text
    cut = min(idxs)
    return text[:cut].strip()
