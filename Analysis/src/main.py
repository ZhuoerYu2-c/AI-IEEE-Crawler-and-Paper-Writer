"""
学术报告自动生成系统 - 主入口

当前版本固定只生成三部分：
1. Introduction
2. Related Work
3. Multiple Viewpoints

以后换主题时，只需要修改：configs/topic.yaml
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from .ingest.chunk import make_chunks
from .ingest.clean import basic_clean, truncate_references
from .ingest.load_texts import load_raw_texts
from .llm.generate_report import generate_full_report
from .rag.build_index import build_faiss_index
from .settings import load_settings


def _copy_papers_to_raw() -> None:
    """自动从 mineru_out 目录复制论文到 data/raw"""
    s = load_settings()
    source_base = s.root.parent / "DownloadPDF" / "Paper_test" / "mineru_out"

    if not source_base.exists():
        print(f"⚠️ 源目录不存在: {source_base}")
        return

    md_files = list(source_base.rglob("txt/*.md"))
    if not md_files:
        print("⚠️ 未找到任何论文文件")
        return

    s.raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📥 发现 {len(md_files)} 篇论文，正在复制...")
    for src in tqdm(md_files, desc="复制论文", unit="篇"):
        dst = s.raw_dir / src.name
        shutil.copy2(src, dst)

    print(f"✅ 已复制 {len(md_files)} 篇论文到 {s.raw_dir}")


def _build_one_index(
    raw_dir: Path,
    cleaned_dir: Path,
    chunks_dir: Path,
    index_dir: Path,
    emb_api_key: str,
    emb_model: str,
    emb_dimensions: int,
    emb_batch_size: int,
    truncate_refs: bool = True,
    label: str = "evidence",
) -> None:
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    raw_items = load_raw_texts(raw_dir)
    if not raw_items:
        print(f"⚠️ {raw_dir} 为空，跳过 {label} 索引构建。")
        return

    all_chunks: List[Dict] = []
    cleaned_items = []

    print("  🧹 清洗文本...")
    for path, text in tqdm(raw_items, desc=f"清洗{label}", unit="篇"):
        cleaned = basic_clean(text)
        if truncate_refs:
            cleaned = truncate_references(cleaned)
        cleaned_items.append((path, cleaned))

        rel = path.relative_to(raw_dir)
        out_clean = cleaned_dir / rel.with_suffix(".txt")
        out_clean.parent.mkdir(parents=True, exist_ok=True)
        out_clean.write_text(cleaned, encoding="utf-8")

    print("  ✂️  文本分块...")
    s = load_settings()
    for path, cleaned in tqdm(cleaned_items, desc=f"分块{label}", unit="篇"):
        rel = path.relative_to(raw_dir)
        chunks = make_chunks(
            source=str(rel).replace("\\", "/"),
            text=cleaned,
            max_chars=s.max_chars,
            overlap=s.overlap_chars,
            min_chars=s.min_chars,
            split_on_headings=s.split_on_headings,
        )

        for c in chunks:
            (chunks_dir / f"{c.chunk_id}.txt").write_text(c.text, encoding="utf-8")
            all_chunks.append(
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "title": c.title,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "text": c.text,
                    "library": label,
                }
            )

    if not all_chunks:
        print(f"⚠️ {label} 切块为空，跳过索引构建。")
        return

    print(f"  📊 共 {len(all_chunks)} 个文本块")
    print("  🔢 生成向量...")
    build_faiss_index(
        chunks=all_chunks,
        index_dir=index_dir,
        emb_api_key=emb_api_key,
        emb_model=emb_model,
        emb_dimensions=emb_dimensions,
        emb_batch_size=emb_batch_size,
    )
    print(f"  ✅ {label} 索引构建完成。")


def build_index_pipeline() -> None:
    s = load_settings()

    print("\n📦 构建证据材料索引...")
    _build_one_index(
        raw_dir=s.raw_dir,
        cleaned_dir=s.cleaned_dir,
        chunks_dir=s.chunks_dir,
        index_dir=s.index_dir,
        emb_api_key=s.qwen_api_key,
        emb_model=s.emb_model,
        emb_dimensions=s.emb_dimensions,
        emb_batch_size=s.emb_batch_size,
        truncate_refs=True,
        label="evidence",
    )

    if s.example_dir.exists() and any(s.example_dir.rglob("*.md")):
        print("\n📦 构建风格参考索引...")
        _build_one_index(
            raw_dir=s.example_dir,
            cleaned_dir=s.example_cleaned_dir,
            chunks_dir=s.example_chunks_dir,
            index_dir=s.style_index_dir,
            emb_api_key=s.qwen_api_key,
            emb_model=s.emb_model,
            emb_dimensions=s.emb_dimensions,
            emb_batch_size=s.emb_batch_size,
            truncate_refs=False,
            label="style",
        )
    else:
        print("\n📦 跳过风格参考索引（目录为空）")


def generate_report_pipeline() -> None:
    print("\n✍️  开始生成三部分报告（Markdown 格式）...")
    s = load_settings()
    generate_full_report(s)


def main() -> None:
    parser = argparse.ArgumentParser(description="学术报告自动生成系统（三部分固定版）")
    parser.add_argument("--skip-copy", action="store_true", help="跳过复制论文步骤")
    parser.add_argument("--skip-index", action="store_true", help="跳过构建索引步骤")
    args = parser.parse_args()

    s = load_settings()

    print("=" * 60)
    print("📚 学术报告自动生成系统")
    print("📌 固定生成：Introduction / Related Work / Multiple Viewpoints")
    print(f"📌 当前主题文件：{s.topic_config_path}")
    print(f"📌 当前主题：{s.paper_topic}")
    print("=" * 60)

    if not args.skip_copy:
        print("\n📥 第一步：复制论文到 data/raw...")
        _copy_papers_to_raw()
    else:
        print("\n📥 跳过复制步骤（--skip-copy）")

    if not args.skip_index:
        print("\n📦 第二步：构建向量索引...")
        build_index_pipeline()
    else:
        print("\n📦 跳过索引构建步骤（--skip-index）")

    print("\n✍️  第三步：生成三部分报告...")
    generate_report_pipeline()

    print("\n" + "=" * 60)
    print("✅ 完成！输出文件在 outputs/ 目录下")
    print("   - 写作计划: outputs/skeleton_sections/")
    print("   - 各部分正文: outputs/markdown_sections/sections/")
    print("   - 引用映射: outputs/markdown_sections/citation_maps/")
    print("   - 检索上下文快照: outputs/markdown_sections/retrieved_context/")
    print("   - 合并结果: outputs/paper.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
