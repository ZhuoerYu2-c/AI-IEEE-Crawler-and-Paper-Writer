"""
报告生成核心模块（固定三部分：Introduction / Related Work / Multiple Viewpoints）

设计目标：
- 主题相关信息只来自 settings 中的 paper_topic / paper_title / paper_keywords / paper_notes
- 代码与 prompt 不携带任何特定学科默认路线
- 以后换主题时，只需修改 configs/topic.yaml
- 正文中的关键判断带有可追踪的 [RefX] 标签
"""
from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..rag.format_context import format_context
from ..rag.retrieve import retrieve
from ..settings import Settings
from .client import MiniMaxClient, stream_chat


REF_PATTERN = re.compile(r"\[(Ref\d+)\]")


def _call_llm(
    client: MiniMaxClient,
    system_prompt: str,
    user_content: str,
    max_tokens: int,
    temperature: float,
    stream: bool = False,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    if stream:
        return stream_chat(
            client=client,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    return client.chat(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=3,
        retry_wait_s=8,
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:latex|markdown|md|json|txt)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.I)
    return text.strip()


def _safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", name).strip()[:80]


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _merge_definitions(old: Dict, new: Dict) -> Dict:
    merged = {"defined_terms": [], "defined_symbols": [], "defined_rules": []}
    for key in merged.keys():
        seen = set()
        items = []
        for src in [old.get(key, []), new.get(key, [])]:
            for x in src:
                if not isinstance(x, str):
                    continue
                xx = x.strip()
                if xx and xx not in seen:
                    seen.add(xx)
                    items.append(xx)
        merged[key] = items
    return merged


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _safe_write(path: Path, content: str) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    if os.path.getsize(temp_path) <= 0:
        raise RuntimeError(f"写入失败，文件为空：{path}")
    temp_path.replace(path)


def _extract_ref_ids(text: str) -> List[str]:
    seen: List[str] = []
    seen_set = set()
    for ref_id in REF_PATTERN.findall(text):
        if ref_id not in seen_set:
            seen_set.add(ref_id)
            seen.append(ref_id)
    return seen


def _normalize_ref_spacing(text: str) -> str:
    text = re.sub(r"\s+(\[Ref\d+\])", r"\1", text)
    text = re.sub(r"(\[Ref\d+\])\s+(\[Ref\d+\])", r"\1\2", text)
    return text


def _build_reference_catalog(reference_records: Sequence[Dict]) -> str:
    if not reference_records:
        return "（无可用引用标签）"

    lines: List[str] = []
    for rec in reference_records:
        preview = (rec.get("text") or "").replace("\n", " ").strip()
        if len(preview) > 160:
            preview = preview[:160] + "..."
        lines.append(
            f"[{rec['ref_id']}] source={rec.get('source')} | title={rec.get('title')} | "
            f"chunk_id={rec.get('chunk_id')} | span=({rec.get('start_char')},{rec.get('end_char')}) | "
            f"score={rec.get('score', 0.0):.4f} | preview={preview}"
        )
    return "\n".join(lines)


def _write_reference_map(path: Path, section_title: str, reference_records: Sequence[Dict], used_ref_ids: Sequence[str]) -> None:
    used_set = set(used_ref_ids)
    lines: List[str] = [f"# {section_title} Citation Map", ""]
    if not reference_records:
        lines.append("（本节无检索证据）")
    else:
        for rec in reference_records:
            flag = "used" if rec["ref_id"] in used_set else "unused"
            preview = (rec.get("text") or "").replace("\n", " ").strip()
            if len(preview) > 300:
                preview = preview[:300] + "..."
            lines.extend(
                [
                    f"## [{rec['ref_id']}] ({flag})",
                    f"- source: {rec.get('source')}",
                    f"- title: {rec.get('title')}",
                    f"- chunk_id: {rec.get('chunk_id')}",
                    f"- span: ({rec.get('start_char')}, {rec.get('end_char')})",
                    f"- score: {rec.get('score', 0.0):.4f}",
                    f"- preview: {preview}",
                    "",
                ]
            )
    _safe_write(path, "\n".join(lines).rstrip() + "\n")


def _build_fixed_section_specs(s: Settings) -> List[Dict[str, str]]:
    topic = s.paper_topic.strip() or "当前主题"
    notes = s.paper_notes.strip()
    extra = f"\n- 补充提示：{notes}" if notes else ""

    return [
        {
            "title": "Introduction",
            "brief": (
                "## Introduction\n"
                "### 本节任务\n"
                f"- 交代“{topic}”的研究背景、问题来源、现实意义与学术价值。\n"
                "- 明确本文的讨论边界、材料来源、分析目标与组织方式。\n"
                "- 说明为什么需要对该主题做系统梳理，并自然引出后续两节。\n"
                "\n"
                "### 必须覆盖的内容\n"
                "- 主题背景与重要性\n"
                "- 核心问题、关键矛盾或研究动因\n"
                "- 为什么现有研究值得系统回顾\n"
                "- 本文后续两节的组织逻辑\n"
                "\n"
                "### 禁止偏离\n"
                "- 不要展开成完整方案设计\n"
                "- 不要写结果讨论、结论展望或无关章节\n"
                "- 不要提前把多个观点完整写完，只做铺垫\n"
                f"{extra}"
            ).strip(),
        },
        {
            "title": "Related Work",
            "brief": (
                "## Related Work\n"
                "### 本节任务\n"
                f"- 围绕“{topic}”系统梳理已有研究。\n"
                "- 对文献进行分类、比较、评述与归纳，而不是逐篇摘要。\n"
                "- 提炼已有研究的共识、分歧、局限与空白，为下一节的多个观点提供依据。\n"
                "\n"
                "### 必须覆盖的内容\n"
                "- 可按研究路径、问题设定、材料类型、方法范式或应用场景分类\n"
                "- 每一类工作的核心思路、适用前提、优势与不足\n"
                "- 不同路线之间的联系、差异与演进关系\n"
                "- 从综述中归纳出后续可展开的观点方向\n"
                "\n"
                "### 禁止偏离\n"
                "- 不要写成逐篇摘录\n"
                "- 不要写成完整新方案\n"
                "- 不要写结果分析或结论章节内容\n"
                f"{extra}"
            ).strip(),
        },
        {
            "title": "Multiple Viewpoints",
            "brief": (
                "## Multiple Viewpoints\n"
                "### 本节任务\n"
                "- 在前两节基础上，提出 3~6 个彼此区分、逻辑清晰的观点。\n"
                "- 每个观点都必须来自前文梳理与比较，而不是凭空想象。\n"
                "- 每个观点都要说明：为什么重要、依据何在、回应了什么研究现象、意味着什么。\n"
                "\n"
                "### 必须覆盖的内容\n"
                "- 用多个小节分别展开不同观点\n"
                "- 每个观点都有明确标题、核心论断与论证说明\n"
                "- 观点之间避免重复，并体现不同侧重点\n"
                "- 结尾可用一小段总结这些观点之间的关系\n"
                "\n"
                "### 禁止偏离\n"
                "- 不要把观点写成完整实施方案\n"
                "- 不要写实验流程、结果分析、结论展望\n"
                "- 不要只列提纲，必须写成学术段落\n"
                f"{extra}"
            ).strip(),
        },
    ]


def _write_fixed_plan_outputs(s: Settings, section_specs: List[Dict[str, str]]) -> None:
    out_root = s.outputs_dir / "skeleton_sections"
    sections_dir = out_root / "sections"
    summaries_dir = out_root / "summaries"

    _reset_dir(out_root)
    sections_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    outline_titles = [spec["title"] for spec in section_specs]
    _safe_write(out_root / "outline.txt", "\n".join(outline_titles) + "\n")

    combined_parts: List[str] = []
    for idx, spec in enumerate(section_specs, 1):
        sec_path = sections_dir / f"{idx:02d}_{_safe_filename(spec['title'])}.md"
        _safe_write(sec_path, spec["brief"].strip() + "\n")
        combined_parts.append(spec["brief"].strip())

    combined = "\n\n".join(combined_parts).strip() + "\n"
    _safe_write(out_root / "writing_skeleton.md", combined)
    _safe_write(s.skeleton_path, combined)


def _build_evidence_query(section_title: str, section_brief: str, previous_summary: str, definitions: Dict, keywords: str) -> str:
    brief = section_brief.replace("\n", " ").strip()
    if len(brief) > 500:
        brief = brief[:500]

    summary = previous_summary.replace("\n", " ").strip()
    if len(summary) > 260:
        summary = summary[:260]

    terms = definitions.get("defined_terms", [])[:8]
    symbols = definitions.get("defined_symbols", [])[:5]

    return f"{section_title}\n{brief}\n{summary}\n{keywords}\n{' '.join(terms)}\n{' '.join(symbols)}".strip()


def _build_style_query(section_title: str, section_brief: str) -> str:
    brief = section_brief.replace("\n", " ").strip()
    if len(brief) > 360:
        brief = brief[:360]
    return f"{section_title}\n{brief}\n中文学术综述写法 段落组织 比较评述 观点提炼".strip()


def _select_diverse_hits(
    hits: Sequence[Dict],
    *,
    max_total: int,
    max_per_source: int,
) -> List[Dict]:
    """
    优先保留高分结果，同时限制同一来源论文占满上下文。
    """
    selected: List[Dict] = []
    source_counts = defaultdict(int)
    seen_chunks = set()

    for hit in hits:
        chunk_id = hit.get("chunk_id")
        source = str(hit.get("source") or "")
        dedupe_key = chunk_id or (
            source,
            hit.get("start_char"),
            hit.get("end_char"),
        )
        if dedupe_key in seen_chunks:
            continue
        if source_counts[source] >= max_per_source:
            continue

        selected.append(hit)
        seen_chunks.add(dedupe_key)
        source_counts[source] += 1

        if len(selected) >= max_total:
            break

    return selected


def summarize_previous(s: Settings, client: MiniMaxClient, section_text: str) -> str:
    prompt = (s.prompts_dir / "summary_prompt.txt").read_text(encoding="utf-8")
    return _call_llm(
        client=client,
        system_prompt=prompt,
        user_content=section_text,
        max_tokens=s.max_tokens_summary,
        temperature=0.2,
    )


def extract_definitions(s: Settings, client: MiniMaxClient, section_text: str) -> Dict:
    prompt = (s.prompts_dir / "definition_extract_prompt.txt").read_text(encoding="utf-8")
    raw = _call_llm(
        client=client,
        system_prompt=prompt,
        user_content=section_text,
        max_tokens=900,
        temperature=0.1,
    ).strip()
    raw = raw.strip("` \n")

    try:
        data = json.loads(raw)
    except Exception:
        data = {"defined_terms": [], "defined_symbols": [], "defined_rules": []}

    for key in ["defined_terms", "defined_symbols", "defined_rules"]:
        if key not in data or not isinstance(data[key], list):
            data[key] = []

    return data


def repeat_check_section(
    s: Settings,
    client: MiniMaxClient,
    section_text: str,
    definitions: Dict,
    reference_records: Sequence[Dict],
) -> str:
    prompt = (s.prompts_dir / "repeat_check_prompt.txt").read_text(encoding="utf-8")
    user_content = (
        "已定义内容表：\n"
        f"{json.dumps(definitions, ensure_ascii=False, indent=2)}\n\n"
        "可用引用标签表：\n"
        f"{_build_reference_catalog(reference_records)}\n\n"
        "当前节 Markdown 正文：\n"
        f"{section_text}"
    )
    checked = _call_llm(
        client=client,
        system_prompt=prompt,
        user_content=user_content,
        max_tokens=s.max_tokens_section,
        temperature=0.1,
    )
    return _normalize_ref_spacing(_strip_fences(_strip_thinking_blocks(checked)))


def generate_one_section(
    s: Settings,
    client: MiniMaxClient,
    section_title: str,
    section_brief: str,
    previous_summary: str,
    definitions: Dict,
) -> Tuple[str, List[Dict], str]:
    evidence_query = _build_evidence_query(
        section_title,
        section_brief,
        previous_summary,
        definitions,
        s.paper_keywords,
    )
    style_query = _build_style_query(section_title, section_brief)

    print("   🔍 检索相关证据...")
    evidence_hits = retrieve(
        query=evidence_query,
        index_dir=s.index_dir,
        emb_api_key=s.qwen_api_key,
        emb_model=s.emb_model,
        emb_dimensions=s.emb_dimensions,
        emb_batch_size=s.emb_batch_size,
        top_k=max(s.top_k * 3, s.top_k),
    )
    evidence_hits = _select_diverse_hits(
        evidence_hits,
        max_total=s.top_k,
        max_per_source=2,
    )
    if evidence_hits:
        evidence_context, reference_records = format_context(
            evidence_hits,
            max_chars=s.context_max_chars,
            ref_prefix="Ref",
        )
    else:
        evidence_context, reference_records = "", []

    print("   🔍 检索风格参考...")
    style_hits = retrieve(
        query=style_query,
        index_dir=s.style_index_dir,
        emb_api_key=s.qwen_api_key,
        emb_model=s.emb_model,
        emb_dimensions=s.emb_dimensions,
        emb_batch_size=s.emb_batch_size,
        top_k=max(s.style_top_k * 2, s.style_top_k),
    )
    style_hits = _select_diverse_hits(
        style_hits,
        max_total=s.style_top_k,
        max_per_source=1,
    )
    if style_hits:
        style_context, _ = format_context(
            style_hits,
            max_chars=s.style_context_max_chars,
            ref_prefix="Style",
        )
    else:
        style_context = ""

    prompt = (s.prompts_dir / "section_prompt.txt").read_text(encoding="utf-8")
    prompt = prompt.replace("{section_title}", section_title)
    prompt = prompt.replace("{paper_topic}", s.paper_topic)

    user_content = (
        f"【主题】{s.paper_topic}\n\n"
        f"【标题】{s.paper_title}\n\n"
        "【当前章节任务说明】（这是本节必须遵循的写作计划）：\n"
        f"{section_brief}\n\n"
        "【前文摘要】（仅用于术语与逻辑一致）：\n"
        f"{previous_summary}\n\n"
        "【已定义内容表】（后文不得重复完整定义）：\n"
        f"{json.dumps(definitions, ensure_ascii=False, indent=2)}\n\n"
        "【证据材料】（正文引用只能使用这里出现的 [RefX] 标签）：\n"
        f"{evidence_context}\n\n"
        "【风格参考材料】（仅借鉴写法，不能当作事实来源，也不能作为正文引用）：\n"
        f"{style_context}\n"
    )

    text = _call_llm(
        client=client,
        system_prompt=prompt,
        user_content=user_content,
        max_tokens=s.max_tokens_section,
        temperature=s.temperature,
        stream=True,
    )
    return _normalize_ref_spacing(_strip_fences(_strip_thinking_blocks(text))), reference_records, evidence_context


def generate_full_report(s: Settings) -> Path:
    client = MiniMaxClient(
        api_key=s.api_key,
        base_url=s.base_url,
        model=s.model,
        timeout_s=max(s.timeout_s, 240),
    )

    out_dir = s.outputs_dir / "markdown_sections"
    _reset_dir(out_dir)
    if (s.outputs_dir / "paper.md").exists():
        (s.outputs_dir / "paper.md").unlink()

    section_specs = _build_fixed_section_specs(s)
    _write_fixed_plan_outputs(s, section_specs)

    sec_dir = out_dir / "sections"
    sum_dir = out_dir / "summaries"
    map_dir = out_dir / "citation_maps"
    ctx_dir = out_dir / "retrieved_context"
    sec_dir.mkdir(parents=True, exist_ok=True)
    sum_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    ctx_dir.mkdir(parents=True, exist_ok=True)

    definitions_path = out_dir / "definitions.json"
    definitions = _load_json(
        definitions_path,
        {"defined_terms": [], "defined_symbols": [], "defined_rules": []},
    )

    sections: List[str] = []
    previous_summary = ""

    print("🗑️  已清空上一次的生成结果，重新开始")
    print("\n📌 当前版本固定只生成 3 个部分：Introduction / Related Work / Multiple Viewpoints")

    print("\n" + "=" * 60)
    print("✍️  开始生成报告正文...")
    print("=" * 60)

    all_used_reference_lines: List[str] = ["# Global Citation Index", ""]

    for idx, spec in enumerate(section_specs, 1):
        title = spec["title"]
        brief = spec["brief"]

        safe_title = _safe_filename(title)
        sec_path = sec_dir / f"{idx:02d}_{safe_title}.md"
        sum_path = sum_dir / f"{idx:02d}_{safe_title}.txt"
        map_path = map_dir / f"{idx:02d}_{safe_title}.md"
        ctx_path = ctx_dir / f"{idx:02d}_{safe_title}.md"

        print(f"\n{'=' * 60}")
        print(f"📄 正文章节 {idx}/{len(section_specs)}: {title}")
        print(f"{'=' * 60}")

        sec_text, reference_records, evidence_context = generate_one_section(
            s=s,
            client=client,
            section_title=title,
            section_brief=brief,
            previous_summary=previous_summary,
            definitions=definitions,
        )

        print("\n🔍 正在检查重复内容与引用标签...")
        sec_text = repeat_check_section(
            s=s,
            client=client,
            section_text=sec_text,
            definitions=definitions,
            reference_records=reference_records,
        )

        used_ref_ids = _extract_ref_ids(sec_text)
        if reference_records and not used_ref_ids:
            raise RuntimeError(f"章节 {title} 未生成任何 [RefX] 引用标签，请检查 prompt 或模型输出。")

        _safe_write(sec_path, sec_text.strip() + "\n")
        print(f"✅ 章节已保存：{sec_path.resolve()}")

        if evidence_context.strip():
            _safe_write(ctx_path, evidence_context.strip() + "\n")
        _write_reference_map(map_path, title, reference_records, used_ref_ids)

        all_used_reference_lines.extend([f"## {title}", ""])
        if used_ref_ids:
            used_set = set(used_ref_ids)
            for rec in reference_records:
                if rec["ref_id"] not in used_set:
                    continue
                all_used_reference_lines.append(
                    f"- [{rec['ref_id']}] source={rec.get('source')} | title={rec.get('title')} | "
                    f"chunk_id={rec.get('chunk_id')} | span=({rec.get('start_char')},{rec.get('end_char')})"
                )
        else:
            all_used_reference_lines.append("- （本节无显式引用标签）")
        all_used_reference_lines.append("")

        previous_summary = summarize_previous(s, client, sec_text)
        _safe_write(sum_path, previous_summary.strip() + "\n")

        new_defs = extract_definitions(s, client, sec_text)
        definitions = _merge_definitions(definitions, new_defs)
        _save_json(definitions_path, definitions)

        sections.append(sec_text.strip())

    _safe_write(out_dir / "citation_index.md", "\n".join(all_used_reference_lines).rstrip() + "\n")

    print("\n" + "=" * 60)
    print("📖 正在组装完整报告...")
    print("=" * 60)

    body_text = "\n\n".join(sections).strip() + "\n"
    body_path = out_dir / "body.md"
    _safe_write(body_path, body_text)
    print(f"✅ 各章节正文已保存：{body_path.resolve()}")

    front_parts = [f"# {s.paper_title}", ""]
    if s.paper_keywords:
        front_parts.extend([f"**Keywords:** {s.paper_keywords}", ""])
    front_parts.extend(["---", "", body_text])
    paper_text = "\n".join(front_parts)

    paper_path = s.outputs_dir / "paper.md"
    _safe_write(paper_path, paper_text)
    print(f"✅ 最终报告已保存：{paper_path.resolve()}")

    return paper_path
