from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

from .settings import load_settings
from .llm.client import MiniMaxClient, stream_chat
from .rag.retrieve import retrieve
from .rag.format_context import format_context


def _safe_read(path: Path, default: str = "") -> str:
    if not path.exists():
        return default
    return path.read_text(encoding="utf-8")


def _safe_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:markdown|md|txt)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def _strip_thinking_blocks(text: str) -> str:
    text = text or ""
    # 去掉 <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.I)
    # 顺手兼容一下别的写法
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.I)
    return text.strip()

def _extract_context_text(formatted) -> str:
    """
    兼容两种情况：
    1) 老版本：format_context(...) -> str
    2) 新版本：format_context(...) -> (context_text, reference_records, ...)
    """
    if isinstance(formatted, str):
        return formatted
    if isinstance(formatted, tuple):
        if len(formatted) > 0:
            return str(formatted[0])
        return ""
    return str(formatted)


def _select_diverse_hits(
    hits: Sequence[Dict],
    *,
    max_total: int,
    max_per_source: int,
) -> List[Dict]:
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


def _retrieve_hits(query: str, *, index_dir: Path, top_k: int) -> List[Dict]:
    s = load_settings()

    if not index_dir.exists():
        return []

    return retrieve(
        query=query,
        index_dir=index_dir,
        emb_api_key=s.qwen_api_key,
        emb_model=s.emb_model,
        emb_dimensions=s.emb_dimensions,
        emb_batch_size=s.emb_batch_size,
        top_k=top_k,
    )


def _retrieve_multi_context(
    queries: List[str],
    *,
    index_dir: Path,
    max_chars_each: int,
    top_k_each: int,
) -> str:
    """
    多轮检索后统一去重与编号，避免不同轮次的 [RefX] 标签冲突。
    """
    merged_hits: List[Dict] = []
    seen_chunks = set()
    valid_queries = 0

    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        valid_queries += 1

        hits = _retrieve_hits(
            query=q,
            index_dir=index_dir,
            top_k=max(top_k_each * 3, top_k_each),
        )
        if not hits:
            continue

        hits = _select_diverse_hits(
            hits,
            max_total=top_k_each,
            max_per_source=2,
        )

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
            seen_chunks.add(dedupe_key)
            merged_hits.append(hit)

    if not merged_hits:
        return ""

    formatted = format_context(
        merged_hits,
        max_chars=max_chars_each * max(valid_queries, 1),
        ref_prefix="Ref",
    )
    return _extract_context_text(formatted)


def _call_llm(system_prompt: str, user_content: str, max_tokens: int, temperature: float) -> str:
    s = load_settings()
    client = MiniMaxClient(
        api_key=s.api_key,
        base_url=s.base_url,
        model=s.model,
        timeout_s=max(getattr(s, "timeout_s", 180), 240),
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    text = stream_chat(
        client=client,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = _strip_fences(text)
    text = _strip_thinking_blocks(text)
    return text



def _shorten(text: str, max_len: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    return text[:max_len]


def polish_paper_to_standard_two_sections(overwrite_paper: bool = True) -> Path:
    """
    两轮后处理：
    第 1 轮：只生成并保存 Section 1（Introduction and Related Work）
    第 2 轮：基于已生成的 Section 1 再次检索，生成 Section 2（Hypotheses and Mathematical Modeling）

    输出：
      - outputs/paper_draft_before_polish.md
      - outputs/paper_intro_related.md
      - outputs/paper_modeling.md
      - outputs/paper_standardized.md
      - outputs/paper.md  （若 overwrite_paper=True）
      - outputs/postprocess_context/
    """
    s = load_settings()

    draft_path = s.outputs_dir / "paper.md"
    if not draft_path.exists():
        raise FileNotFoundError(
            f"未找到初稿文件：{draft_path}\n"
            "请先完成三段式生成，确保 outputs/paper.md 已存在。"
        )

    draft_text = _safe_read(draft_path).strip()
    if not draft_text:
        raise RuntimeError(f"初稿文件为空：{draft_path}")

    citation_index_path = s.outputs_dir / "markdown_sections" / "citation_index.md"
    citation_index_text = _safe_read(citation_index_path, "")

    debug_dir = s.outputs_dir / "postprocess_context"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 第一轮：只为 Introduction and Related Work 做检索与润色
    # ============================================================
    query_intro_1 = (
        f"{s.paper_topic}\n"
        "research background significance motivation problem setting literature review "
        "related work taxonomy comparison gap development trend"
    )

    query_intro_2 = (
        f"{s.paper_topic}\n"
        "state of the art survey categories key approaches strengths weaknesses limitations"
    )

    intro_context = _retrieve_multi_context(
        queries=[query_intro_1, query_intro_2],
        index_dir=s.index_dir,
        max_chars_each=min(getattr(s, "context_max_chars", 12000), 9000),
        top_k_each=max(6, min(getattr(s, "top_k", 8), 10)),
    )

    style_context = ""
    if s.style_index_dir.exists():
        try:
            style_context = _retrieve_multi_context(
                queries=[
                    "academic introduction related work review formal Chinese writing logical transition",
                    "survey writing literature review comparison gap motivation formal academic style",
                ],
                index_dir=s.style_index_dir,
                max_chars_each=min(getattr(s, "style_context_max_chars", 8000), 6000),
                top_k_each=max(4, min(getattr(s, "style_top_k", 6), 8)),
            )
        except Exception:
            style_context = ""

    _safe_write(debug_dir / "01_input_draft.txt", draft_text or "")
    _safe_write(debug_dir / "02_citation_index.txt", citation_index_text or "")
    _safe_write(debug_dir / "03_intro_context.txt", intro_context or "")
    _safe_write(debug_dir / "04_style_context.txt", style_context or "")

    system_prompt_intro = """
你是一名中文学术论文写作助手。你当前只负责生成论文的第一部分，而不是整篇论文。

你的任务是：
将输入草稿与知识库证据整合、润色为规范的第一部分：

## 1 Introduction and Related Work

你必须严格满足以下要求：

【结构要求】
1. 只输出这一部分，不要输出第二部分。
2. 顶层标题必须是：
   ## 1 Introduction and Related Work
3. 允许在其下使用多个 ### 小节。

【内容要求】
这一部分必须完成：
1. 研究背景、问题场景与研究意义；
2. 说明该主题为何值得系统研究；
3. 对已有研究进行分类综述与比较分析，而不是简单罗列；
4. 提炼已有研究的共识、分歧、局限与空白；
5. 在本节结尾自然引出：为什么下一步需要提出假设并进行数学建模。

【特别要求】
1. 不能编造知识库中没有支撑的事实与结论；
2. 不要写具体实验结果、仿真参数或完整技术方案；
3. 不要把“建模”提前展开成第二节内容，只做逻辑引出；
4. 如果证据中含有 [RefX]，关键判断、分类比较与研究空白后必须尽量保留，且不要只集中重复极少数标签；
5. 输出必须是完整、正式、自然、连贯的学术中文；
6. 只输出最终 Markdown 正文，不要解释过程，不要输出代码块。
""".strip()

    user_content_intro = f"""
【当前研究主题】
{s.paper_topic}

【现有初稿（用于整合和重写）】
{draft_text}

【知识库证据：用于写第一部分】
{intro_context}

【已有引用索引】
{citation_index_text}

【风格参考（仅借鉴写法，不当事实来源）】
{style_context}
""".strip()

    section1 = _call_llm(
        system_prompt=system_prompt_intro,
        user_content=user_content_intro,
        max_tokens=max(getattr(s, "max_tokens_section", 6000), 3800),
        temperature=min(getattr(s, "temperature", 0.7), 0.35),
    ).strip()

    if not section1:
        raise RuntimeError("第一轮润色失败：Section 1 返回空文本。")

    if "## 1 " not in section1 and "## 1." not in section1:
        print("⚠️ 第一轮输出中未检测到 '## 1 ...' 顶层标题，请人工检查结构。")

    section1_path = s.outputs_dir / "paper_intro_related.md"
    _safe_write(section1_path, section1.strip() + "\n")
    _safe_write(debug_dir / "05_section1_polished.txt", section1 or "")

    # ============================================================
    # 第二轮：基于第一轮结果，再次索引建模相关证据
    # ============================================================
    section1_tail = _shorten(section1, 1800)
    topic_short = _shorten(s.paper_topic, 300)

    query_model_1 = (
        f"{topic_short}\n"
        f"{section1_tail}\n"
        "hypothesis assumptions mathematical modeling variable notation problem formulation "
        "objective constraint abstract formalization"
    )

    query_model_2 = (
        f"{topic_short}\n"
        "system model variable definition assumptions objective function constraints "
        "mathematical formulation abstract modeling"
    )

    query_model_3 = (
        f"{topic_short}\n"
        "based on literature gaps formulate hypotheses and abstract mathematical model"
    )

    model_context = _retrieve_multi_context(
        queries=[query_model_1, query_model_2, query_model_3],
        index_dir=s.index_dir,
        max_chars_each=min(getattr(s, "context_max_chars", 12000), 7000),
        top_k_each=max(6, min(getattr(s, "top_k", 8), 10)),
    )

    _safe_write(debug_dir / "06_model_context.txt", model_context or "")

    system_prompt_model = """
你是一名中文学术论文写作助手。你当前只负责生成论文的第二部分，而不是整篇论文。

你的任务是：
基于已经完成的第一部分，以及新检索到的建模相关证据，生成规范的第二部分：

## 2 Hypotheses and Mathematical Modeling

你必须严格满足以下要求：

【结构要求】
1. 只输出这一部分，不要重复输出第一部分。
2. 顶层标题必须是：
   ## 2 Hypotheses and Mathematical Modeling
3. 允许在其下使用多个 ### 小节。

【逻辑要求】
1. 第二部分必须明确承接第一部分的结尾与研究空白；
2. 开头要用一小段话说明：为何基于前述综述，需要提出假设并建立数学建模；
3. 整节必须像同一篇论文自然接下来的内容，不能各说各话。

【内容要求】
这一部分必须完成：
1. 基于第一部分归纳的问题，提出一组清晰、克制、合理的研究假设；
2. 在假设之上建立抽象的数学建模表达；
3. 明确对象、变量、符号、关系、目标或约束；
4. 如果知识库不足以支撑很具体的公式，就使用抽象数学形式与变量定义；
5. 可以给出适度的数学符号，但不能编造过细、过实、无依据的公式或技术细节；
6. 数学建模必须与第一部分综述中提出的问题保持一致。

【特别限制】
1. 不能凭空给出实验、仿真、参数或结果；
2. 不能伪造完整算法流程；
3. 不要和第一部分重复大段综述内容；
4. 如果证据中含有 [RefX]，关键判断、假设依据与建模抽象后必须尽量保留，且不要只集中重复极少数标签；
5. 输出必须是正式、自然、连贯的学术中文；
6. 只输出最终 Markdown 正文，不要解释过程，不要输出代码块。
""".strip()

    user_content_model = f"""
【当前研究主题】
{s.paper_topic}

【已经完成的第一部分（你必须承接它）】
{section1}

【原始初稿（仅作补充参考，不可凌驾于第一部分之上）】
{draft_text}

【建模相关知识库证据（这是第二轮重新检索得到的）】
{model_context}

【已有引用索引】
{citation_index_text}

【风格参考（仅借鉴写法，不当事实来源）】
{style_context}
""".strip()

    section2 = _call_llm(
        system_prompt=system_prompt_model,
        user_content=user_content_model,
        max_tokens=max(getattr(s, "max_tokens_section", 6000), 3200),
        temperature=min(getattr(s, "temperature", 0.7), 0.3),
    ).strip()

    if not section2:
        raise RuntimeError("第二轮润色失败：Section 2 返回空文本。")

    if "## 2 " not in section2 and "## 2." not in section2:
        print("⚠️ 第二轮输出中未检测到 '## 2 ...' 顶层标题，请人工检查结构。")

    section2_path = s.outputs_dir / "paper_modeling.md"
    _safe_write(section2_path, section2.strip() + "\n")
    _safe_write(debug_dir / "07_section2_polished.txt", section2 or "")

    # ============================================================
    # 合并最终稿
    # ============================================================
    final_text = section1.strip() + "\n\n" + section2.strip() + "\n"

    backup_path = s.outputs_dir / "paper_draft_before_polish.md"
    standardized_path = s.outputs_dir / "paper_standardized.md"

    _safe_write(backup_path, draft_text.strip() + "\n")
    _safe_write(standardized_path, final_text)

    if overwrite_paper:
        _safe_write(draft_path, final_text)

    return standardized_path


if __name__ == "__main__":
    out = polish_paper_to_standard_two_sections(overwrite_paper=True)
    print("=" * 60)
    print("✅ 两轮标准化润色完成")
    print(f"输出文件：{out}")
    print("同时生成：")
    print(" - outputs/paper_intro_related.md")
    print(" - outputs/paper_modeling.md")
    print(" - outputs/paper_standardized.md")
    print(" - outputs/paper.md")
    print("=" * 60)
