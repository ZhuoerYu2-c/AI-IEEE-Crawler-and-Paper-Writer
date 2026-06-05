from __future__ import annotations

import sys
import json
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import faiss


# ============================================================
# 让 test.py 能在“和主入口同目录”下直接运行
# ============================================================
CUR_DIR = Path(__file__).resolve().parent
PKG_NAME = CUR_DIR.name
PARENT_DIR = CUR_DIR.parent

if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))


# ============================================================
# 导入你项目现有模块
# ============================================================
settings_mod = __import__(f"{PKG_NAME}.settings", fromlist=["*"])
build_mod = __import__(f"{PKG_NAME}.rag.build_index", fromlist=["*"])

load_settings = settings_mod.load_settings


# ============================================================
# 你这次学习 RAG 的检索 query
# 这里只是“检索语句”，不是最终发给大模型的完整 prompt
# ============================================================
QUERY = """
fuzzy control, fuzzy neural network, Takagi-Sugeno fuzzy model,
adaptive fuzzy control, fuzzy sliding mode control,
disturbance observer, nonlinear system, stability analysis
""".strip()

TOP_K = 5


# ============================================================
# 输出目录
# ============================================================
DEBUG_DIR = CUR_DIR / "outputs" / "debug_rag_learning"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def print_title(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def short_text(text: str, limit: int = 1500) -> str:
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [已截断，原长度={len(s)}]"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_meta_jsonl(meta_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"meta.jsonl 第 {line_no} 行解析失败: {e}")
    return rows


def detect_index_and_meta(s) -> tuple[Path, Path]:
    """
    优先用 settings 里的 index_dir；
    找不到再回退到你日志里那条固定路径。
    """
    candidates = []

    if hasattr(s, "index_dir"):
        index_dir = Path(s.index_dir)
        candidates.append((index_dir / "faiss.index", index_dir / "meta.jsonl"))

    candidates.append((
        Path("/Users/joe/Downloads/codex/Smart-Manufacturing-Competition/Yu/Analysis/index/faiss.index"),
        Path("/Users/joe/Downloads/codex/Smart-Manufacturing-Competition/Yu/Analysis/index/meta.jsonl"),
    ))

    for index_path, meta_path in candidates:
        if index_path.exists() and meta_path.exists():
            return index_path, meta_path

    raise FileNotFoundError("没找到 faiss.index 和 meta.jsonl，请确认索引确实已构建完成。")


def find_embedding_function(module) -> Callable[..., Any]:
    """
    自动找 rag.build_index 里负责 embedding 的函数，
    这样就尽量复用你项目原本的 embedding 调用逻辑。
    """
    candidates = []

    for name, obj in vars(module).items():
        if callable(obj) and "embed" in name.lower():
            try:
                sig = inspect.signature(obj)
                params = set(sig.parameters.keys())
                score = 0

                if "texts" in params:
                    score += 3
                if "input_texts" in params:
                    score += 3
                if "api_key" in params or "emb_api_key" in params:
                    score += 2
                if "model" in params or "emb_model" in params:
                    score += 2
                if "dimensions" in params or "emb_dimensions" in params:
                    score += 1
                if "batch_size" in params or "emb_batch_size" in params:
                    score += 1

                candidates.append((score, name, obj, sig))
            except Exception:
                candidates.append((0, name, obj, "(unknown)"))

    if not candidates:
        raise RuntimeError("在 rag.build_index 中没有找到包含 embed 的函数。")

    candidates.sort(key=lambda x: x[0], reverse=True)

    print_title("自动识别到的 embedding 函数")
    for score, name, _, sig in candidates:
        print(f"- {name}{sig}    [score={score}]")

    best = candidates[0]
    print(f"\n✅ 选中：{best[1]}{best[3]}")
    return best[2]


def call_embedding_function(embed_func: Callable[..., Any], texts: List[str], s) -> np.ndarray:
    """
    用项目现有 embedding 函数把 query 变成向量。
    注意：这里只会调用 embedding，不会调用大模型聊天生成。
    """
    sig = inspect.signature(embed_func)
    params = sig.parameters
    kwargs = {}

    # 文本参数
    if "texts" in params:
        kwargs["texts"] = texts
    elif "input_texts" in params:
        kwargs["input_texts"] = texts
    elif "inputs" in params:
        kwargs["inputs"] = texts
    elif "text_list" in params:
        kwargs["text_list"] = texts
    else:
        # 兜底：尝试单参数直接传入
        try:
            out = embed_func(texts)
            arr = np.asarray(out, dtype="float32")
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception as e:
            raise RuntimeError(f"embedding 函数调用失败：{e}")

    # API key / model / dimensions / batch size
    if "emb_api_key" in params:
        kwargs["emb_api_key"] = s.qwen_api_key
    if "api_key" in params:
        kwargs["api_key"] = s.qwen_api_key

    if "emb_model" in params:
        kwargs["emb_model"] = s.emb_model
    if "model" in params:
        kwargs["model"] = s.emb_model

    if "emb_dimensions" in params:
        kwargs["emb_dimensions"] = s.emb_dimensions
    if "dimensions" in params:
        kwargs["dimensions"] = s.emb_dimensions

    if "emb_batch_size" in params:
        kwargs["emb_batch_size"] = s.emb_batch_size
    if "batch_size" in params:
        kwargs["batch_size"] = s.emb_batch_size

    out = embed_func(**kwargs)
    arr = np.asarray(out, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def detect_similarity_hint() -> Dict[str, bool]:
    """
    粗略看 build_index 源码里是不是：
    - 用了内积 IndexFlatIP
    - 对向量做了 L2 normalize
    这样 query 检索时可以尽量一致。
    """
    src = inspect.getsource(build_mod)
    use_ip = ("IndexFlatIP" in src) or ("METRIC_INNER_PRODUCT" in src)
    use_norm = ("normalize_L2" in src) or ("faiss.normalize_L2" in src)
    return {"use_ip": use_ip, "use_norm": use_norm}


def retrieve(index_path: Path, meta_rows: List[Dict], qvec: np.ndarray, top_k: int) -> List[Dict]:
    index = faiss.read_index(str(index_path))
    scores, ids = index.search(qvec, top_k)

    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[Dict] = []
    for rank, (score, idx) in enumerate(zip(scores, ids), start=1):
        if idx < 0 or idx >= len(meta_rows):
            continue

        row = meta_rows[idx]
        results.append({
            "rank": rank,
            "score": float(score),
            "faiss_id": int(idx),
            "chunk_id": row.get("chunk_id"),
            "source": row.get("source"),
            "title": row.get("title"),
            "start_char": row.get("start_char"),
            "end_char": row.get("end_char"),
            "text": row.get("text", ""),
            "library": row.get("library"),
        })

    return results


def build_prompt_preview(query: str, retrieved: List[Dict]) -> List[Dict]:
    """
    这里只是展示“如果下一步要发给大模型，prompt 大概会怎么拼”。
    不真的调用 LLM。
    """
    context_blocks = []
    for item in retrieved:
        block = (
            f"[Top{item['rank']}] "
            f"source={item['source']} | "
            f"chunk_id={item['chunk_id']} | "
            f"score={item['score']:.4f}\n"
            f"{item['text']}"
        )
        context_blocks.append(block)

    retrieved_context = "\n\n" + ("\n" + "-" * 80 + "\n").join(context_blocks)

    messages = [
        {
            "role": "system",
            "content": (
                "你是学术综述写作助手。"
                "请优先依据给定的检索材料回答，不要脱离材料随意发挥。"
                "如果材料不足，请明确说明材料不足。"
            ),
        },
        {
            "role": "user",
            "content": (
                "下面给你两部分内容：\n"
                "1）我的检索 query\n"
                "2）FAISS 检索回来的论文片段\n\n"
                "请注意：这里还没有真的调用大模型，我只是展示最终可能发送给大模型的 prompt 结构。\n\n"
                f"【检索 query】\n{query}\n\n"
                f"【retrieved context】{retrieved_context}\n\n"
                "【任务】\n"
                "请基于以上检索片段，总结模糊控制研究中常见的方法、研究对象、问题设置以及稳定性分析思路。"
            ),
        },
    ]
    return messages


def main() -> None:
    s = load_settings()

    print_title("当前配置")
    print(f"paper_topic    = {getattr(s, 'paper_topic', 'N/A')}")
    print(f"index_dir      = {getattr(s, 'index_dir', 'N/A')}")
    print(f"emb_model      = {getattr(s, 'emb_model', 'N/A')}")
    print(f"emb_dimensions = {getattr(s, 'emb_dimensions', 'N/A')}")
    print(f"emb_batch_size = {getattr(s, 'emb_batch_size', 'N/A')}")

    index_path, meta_path = detect_index_and_meta(s)

    print_title("读取现有索引")
    print(f"faiss.index = {index_path}")
    print(f"meta.jsonl  = {meta_path}")

    meta_rows = load_meta_jsonl(meta_path)
    print(f"meta 行数 = {len(meta_rows)}")

    print_title("你这次送进 FAISS 的检索 query")
    print(QUERY)

    # 用和建库时同一套 embedding 逻辑生成 query 向量
    embed_func = find_embedding_function(build_mod)
    qvec = call_embedding_function(embed_func, [QUERY], s).astype("float32")

    hint = detect_similarity_hint()
    print_title("索引相似度方式判断")
    print(hint)

    if hint["use_norm"]:
        faiss.normalize_L2(qvec)
        print("✅ 已对 query 向量做 L2 normalize（和建库保持一致）")

    retrieved = retrieve(
        index_path=index_path,
        meta_rows=meta_rows,
        qvec=qvec,
        top_k=TOP_K,
    )

    print_title(f"FAISS 返回的 Top-{len(retrieved)} 个 chunks")
    for item in retrieved:
        print(f"[Top{item['rank']}] score={item['score']:.6f}")
        print(f"faiss_id  : {item['faiss_id']}")
        print(f"source    : {item['source']}")
        print(f"chunk_id  : {item['chunk_id']}")
        print(f"title     : {item['title']}")
        print(f"char_span : {item['start_char']} ~ {item['end_char']}")
        print("-" * 80)
        print(short_text(item["text"], limit=1800))
        print("-" * 80)

    prompt_preview = build_prompt_preview(QUERY, retrieved)

    print_title("最后如果要发给大模型，prompt 大概会这样拼（这里只展示，不发送）")
    print(json.dumps(prompt_preview, ensure_ascii=False, indent=2))

    # 保存学习结果
    save_text(DEBUG_DIR / "query.txt", QUERY)
    save_json(DEBUG_DIR / "retrieved_chunks.json", retrieved)
    save_json(DEBUG_DIR / "prompt_preview.json", prompt_preview)

    summary = f"""
RAG 学习结论：

1. 先送进 embedding / FAISS 的，是“检索 query”：
{QUERY}

2. FAISS 返回的是最相近的若干 chunk，而不是直接生成答案。

3. 真正发给大模型时，通常会把：
   - system 指令
   - user 任务描述
   - retrieved chunks
   拼成一个更长的 prompt。

4. 所以关系是：
   query
   -> 向量化
   -> FAISS 检索出相关 chunks
   -> chunks 拼进 prompt
   -> 再发送给大模型

你现在最该看的两个文件：
- {DEBUG_DIR / "retrieved_chunks.json"}
- {DEBUG_DIR / "prompt_preview.json"}
""".strip()

    print_title("关系总结")
    print(summary)
    save_text(DEBUG_DIR / "summary.txt", summary)

    print_title("完成")
    print(f"输出目录：{DEBUG_DIR}")


if __name__ == "__main__":
    main()