from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    root: Path

    # paths
    raw_dir: Path
    example_dir: Path
    cleaned_dir: Path
    example_cleaned_dir: Path
    chunks_dir: Path
    example_chunks_dir: Path
    index_dir: Path
    style_index_dir: Path
    outputs_dir: Path
    prompts_dir: Path
    skeleton_path: Path
    topic_config_path: Path

    # chunking
    max_chars: int
    overlap_chars: int
    min_chars: int
    split_on_headings: bool

    # retrieval
    top_k: int
    style_top_k: int
    context_max_chars: int
    style_context_max_chars: int

    # topic config
    paper_topic: str
    paper_title: str
    paper_keywords: str
    paper_notes: str
    min_section_chars: int

    # embedding
    emb_model: str
    emb_dimensions: int
    emb_batch_size: int
    qwen_api_key: str

    # llm
    api_key: str
    base_url: str
    model: str
    temperature: float
    max_tokens_section: int
    max_tokens_summary: int
    timeout_s: int


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"❌ 配置文件不存在：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def load_settings() -> Settings:
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "configs" / "config.yaml"
    cfg = _load_yaml(cfg_path)

    paths = cfg["paths"]
    topic_cfg_path = root / paths.get("topic_config_path", "configs/topic.yaml")
    topic_cfg = _load_yaml(topic_cfg_path).get("topic", {})

    chunking = cfg["chunking"]
    retrieval = cfg["retrieval"]
    paper = cfg.get("paper", {})
    llm = cfg["llm"]
    emb = cfg["embedding"]

    qwen_api_key = os.getenv("QWEN_API_KEY", "").strip()
    if not qwen_api_key:
        raise RuntimeError("❌ 未检测到 QWEN_API_KEY，请在 .env 中配置。")

    api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("❌ 未检测到 MINIMAX_API_KEY，请在 .env 中配置。")

    paper_topic = str(topic_cfg.get("subject", "")).strip()
    paper_title = str(topic_cfg.get("title", "")).strip()
    paper_keywords = str(topic_cfg.get("keywords", "")).strip()
    paper_notes = str(topic_cfg.get("notes", "")).strip()

    if not paper_topic:
        raise RuntimeError(f"❌ 主题文件缺少 topic.subject：{topic_cfg_path}")
    if not paper_title:
        paper_title = paper_topic

    return Settings(
        root=root,
        raw_dir=root / paths["raw_dir"],
        example_dir=root / paths["example_dir"],
        cleaned_dir=root / paths["cleaned_dir"],
        example_cleaned_dir=root / paths["example_cleaned_dir"],
        chunks_dir=root / paths["chunks_dir"],
        example_chunks_dir=root / paths["example_chunks_dir"],
        index_dir=root / paths["index_dir"],
        style_index_dir=root / paths["style_index_dir"],
        outputs_dir=root / paths["outputs_dir"],
        prompts_dir=root / "prompts",
        skeleton_path=root / paths["skeleton_path"],
        topic_config_path=topic_cfg_path,
        max_chars=int(chunking["max_chars"]),
        overlap_chars=int(chunking["overlap_chars"]),
        min_chars=int(chunking["min_chars"]),
        split_on_headings=bool(chunking["split_on_headings"]),
        top_k=int(retrieval["top_k"]),
        style_top_k=int(retrieval.get("style_top_k", 4)),
        context_max_chars=int(retrieval["context_max_chars"]),
        style_context_max_chars=int(retrieval.get("style_context_max_chars", 5000)),
        paper_topic=paper_topic,
        paper_title=paper_title,
        paper_keywords=paper_keywords,
        paper_notes=paper_notes,
        min_section_chars=int(paper.get("min_section_chars", 900)),
        emb_model=str(emb["model"]),
        emb_dimensions=int(emb.get("dimensions", 1024)),
        emb_batch_size=int(emb.get("batch_size", 10)),
        qwen_api_key=qwen_api_key,
        api_key=api_key,
        base_url=str(llm["base_url"]),
        model=str(llm["model"]),
        temperature=float(llm["temperature"]),
        max_tokens_section=int(llm["max_tokens_section"]),
        max_tokens_summary=int(llm["max_tokens_summary"]),
        timeout_s=int(llm["timeout_s"]),
    )
