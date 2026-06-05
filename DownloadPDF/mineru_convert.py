import os
import time
import json
import zipfile
import shutil
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# =========================================================
# 0) 环境打印：方便以后继续补 yml
# =========================================================
def print_environment_info():
    import sys

    print("=" * 80)
    print("Python 可执行文件:", sys.executable)
    print("Python 版本:", sys.version.replace("\n", " "))
    print("当前工作目录:", os.getcwd())
    print("Path.cwd():", Path.cwd())
    print("=" * 80)

    packages = [
        ("requests", "requests"),
        ("pandas", "pandas"),
        ("selenium", "selenium"),
        ("webdriver-manager", "webdriver_manager"),
        ("jupyter", "jupyter"),
        ("ipykernel", "ipykernel"),
        ("ipywidgets", "ipywidgets"),
        ("nbformat", "nbformat"),
        ("nbconvert", "nbconvert"),
        ("PyPDF2", "PyPDF2"),
        ("python-docx", "docx"),
        ("openai", "openai"),
        ("tiktoken", "tiktoken"),
        ("httpx", "httpx"),
        ("faiss-cpu", "faiss"),
    ]

    print(f"{'安装包名':<22}{'导入名':<22}{'版本':<18}{'状态'}")
    print("-" * 90)

    for pip_name, import_name in packages:
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "UNKNOWN")
            status = "OK"
        except Exception as e:
            version = "NOT INSTALLED"
            status = f"{type(e).__name__}: {e}"

        print(f"{pip_name:<22}{import_name:<22}{version:<18}{status}")


# =========================================================
# 1) 开关区：你只改这里
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT.parent / ".env")

PDF_DIR = PROJECT_ROOT / "Paper_test" / "pdfs"
OUT_BASE = PROJECT_ROOT / "Paper_test" / "mineru_out"

START_N = 1
END_N = 2

OUTPUT_FORMAT = "md"        # "md" 或 "latex"
MODEL_VERSION = "pipeline"  # "pipeline" 或 "vlm"

ENABLE_FORMULA = True
ENABLE_TABLE = True
LANGUAGE = "ch"

POLL_INTERVAL = 3
POLL_TIMEOUT_SEC = 1800

# 是否在转换前，把 pdfs 目录下的 PDF 重命名为 1.pdf, 2.pdf, 3.pdf, ...
RENAME_PDFS_BEFORE_RUN = True


# =========================================================
# 2) 基础配置
# =========================================================
TOKEN = os.getenv("MINERU_TOKEN", "").strip()
if not TOKEN:
    raise RuntimeError("TOKEN 为空，请在 .env 中配置 MINERU_TOKEN。")

BASE = "https://mineru.net"
H_JSON = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {TOKEN}"
}


def die(msg):
    raise RuntimeError(msg)


def ensure_dirs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    OUT_BASE.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3) 先把现有 PDF 重命名成 1.pdf, 2.pdf, 3.pdf ...
# =========================================================
def rename_pdfs_sequentially(pdf_dir: Path):
    pdfs = sorted(p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")

    if not pdfs:
        die(f"目录里没有 PDF 文件：{pdf_dir}")

    print("\n" + "=" * 80)
    print("[rename] 开始顺序重命名 PDF ...")
    print("[rename] 目录：", pdf_dir)

    # 为避免直接改名时重名冲突，先统一改成临时名
    temp_pairs = []
    for i, p in enumerate(pdfs, start=1):
        tmp_path = p.with_name(f"__tmp_rename_{i}__.pdf")
        if tmp_path.exists():
            tmp_path.unlink()
        p.rename(tmp_path)
        temp_pairs.append((i, tmp_path))

    # 再从临时名改成最终名
    final_files = []
    for i, tmp_path in temp_pairs:
        new_path = pdf_dir / f"{i}.pdf"
        if new_path.exists():
            new_path.unlink()
        tmp_path.rename(new_path)
        final_files.append(new_path)
        print(f"[rename] {tmp_path.name} -> {new_path.name}")

    print("[rename] ✅ 重命名完成")
    return final_files


# =========================================================
# 4) 构造待处理 PDF 列表
# =========================================================
def build_pdf_list(start_n, end_n):
    files = []
    for i in range(start_n, end_n + 1):
        p = PDF_DIR / f"{i}.pdf"
        if not p.exists():
            die(f"文件不存在：{p}")
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"[check] {p.name} size={size_mb:.2f}MB")
        files.append((i, p))
    return files


# =========================================================
# 5) 申请上传链接
# =========================================================
def apply_upload_urls(indexed_files):
    url = f"{BASE}/api/v4/file-urls/batch"

    payload = {
        "files": [{"name": p.name, "data_id": str(i)} for i, p in indexed_files],
        "model_version": MODEL_VERSION,
    }

    if MODEL_VERSION == "pipeline":
        payload["enable_formula"] = ENABLE_FORMULA
        payload["enable_table"] = ENABLE_TABLE
        payload["language"] = LANGUAGE

    if OUTPUT_FORMAT.lower() == "latex":
        payload["extra_formats"] = ["latex"]

    r = requests.post(url, headers=H_JSON, json=payload, timeout=60)
    j = r.json()

    print("[apply] HTTP:", r.status_code)
    print("[apply] resp:", json.dumps(j, ensure_ascii=False, indent=2)[:2000])

    if j.get("code") != 0:
        die(
            f"申请上传URL失败：code={j.get('code')} "
            f"msg={j.get('msg')} trace_id={j.get('trace_id')}"
        )

    batch_id = j["data"]["batch_id"]
    file_urls = j["data"]["file_urls"]

    if len(file_urls) != len(indexed_files):
        die(f"file_urls数量不匹配：期望{len(indexed_files)}，实际{len(file_urls)}")

    return batch_id, file_urls


# =========================================================
# 6) PUT 上传 PDF
# =========================================================
def upload_put(indexed_files, file_urls):
    for (i, p), u in zip(indexed_files, file_urls):
        print(f"[upload] PUT {p.name} (id={i})")
        with open(p, "rb") as f:
            r = requests.put(u, data=f, timeout=600)

        print(f"[upload] {p.name} HTTP:", r.status_code)
        if not (200 <= r.status_code < 300):
            die(f"上传失败：{p.name} HTTP {r.status_code} text={r.text[:200]}")

    print("[upload] ✅ 上传完成（系统会自动提交解析任务）")


# =========================================================
# 7) 轮询解析结果
# =========================================================
def poll_batch(batch_id, target_names, interval=3, timeout_sec=1200):
    url = f"{BASE}/api/v4/extract-results/batch/{batch_id}"
    t0 = time.time()
    done = {name: None for name in target_names}

    while True:
        r = requests.get(url, headers=H_JSON, timeout=60)
        j = r.json()

        if j.get("code") != 0:
            die(
                f"查询失败：code={j.get('code')} "
                f"msg={j.get('msg')} trace_id={j.get('trace_id')}"
            )

        results = j["data"]["extract_result"]
        by_name = {it.get("file_name"): it for it in results if it.get("file_name")}

        for name in target_names:
            it = by_name.get(name)
            if not it:
                continue

            state = it.get("state")
            prog = it.get("extract_progress") or {}
            ep = prog.get("extracted_pages")
            tp = prog.get("total_pages")

            print(f"[poll] {name}: state={state} pages={ep}/{tp}")

            if state in ("done", "failed"):
                done[name] = it

        if all(done.values()):
            return done

        if time.time() - t0 > timeout_sec:
            die(f"轮询超时（>{timeout_sec}s） batch_id={batch_id}")

        time.sleep(interval)


# =========================================================
# 8) 导出主文件
# =========================================================
def pick_largest_file(paths):
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_size)


def download_unzip_and_export(i: int, item: dict):
    fname = item.get("file_name", f"{i}.pdf")
    state = item.get("state")

    out_root = OUT_BASE / str(i) / str(i)
    txt_dir = out_root / "txt"
    out_root.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    if state == "failed":
        print(f"[final] {fname} FAILED: {item.get('err_msg')}")
        return None

    zurl = item.get("full_zip_url")
    if not zurl:
        print(f"[final] {fname} done but no full_zip_url")
        return None

    zip_path = out_root / f"{i}.zip"
    extract_dir = out_root

    print(f"[download] {fname} -> {zip_path}")
    with requests.get(zurl, stream=True, timeout=600) as r:
        if not (200 <= r.status_code < 300):
            die(f"下载失败：{fname} HTTP {r.status_code}")

        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
        names = z.namelist()

    print(f"[unzip] -> {extract_dir}")
    print("[unzip] top files:")
    for n in names[:30]:
        print("  -", n)

    want = OUTPUT_FORMAT.lower()

    if want == "latex":
        candidates = sorted(extract_dir.rglob("*.tex"))
        main = pick_largest_file(candidates)

        if not main:
            print("[export] ⚠️ 没找到 .tex，尝试兜底导出 md")
            md_main = pick_largest_file(sorted(extract_dir.rglob("*.md")))
            if md_main:
                out_path = txt_dir / f"{i}.md"
                shutil.copy2(md_main, out_path)
                print("✅ 兜底导出:", out_path)
                return str(out_path)
            return None

        out_path = txt_dir / f"{i}.tex"
        shutil.copy2(main, out_path)
        print("✅ 导出主LaTeX:", out_path, "| source:", main)
        return str(out_path)

    else:
        candidates = sorted(extract_dir.rglob("*.md"))
        main = pick_largest_file(candidates)

        if not main:
            print("[export] ⚠️ 没找到 .md")
            return None

        out_path = txt_dir / f"{i}.md"
        shutil.copy2(main, out_path)
        print("✅ 导出主Markdown:", out_path, "| source:", main)
        return str(out_path)


# =========================================================
# 9) 主流程
# =========================================================
def main():
    ensure_dirs()
    print_environment_info()

    print("\n" + "=" * 80)
    print("PROJECT_ROOT =", PROJECT_ROOT)
    print("PDF_DIR      =", PDF_DIR)
    print("OUT_BASE     =", OUT_BASE)
    print("START_N      =", START_N)
    print("END_N        =", END_N)
    print("OUTPUT_FORMAT=", OUTPUT_FORMAT)
    print("MODEL_VERSION=", MODEL_VERSION)
    print("RENAME_PDFS_BEFORE_RUN =", RENAME_PDFS_BEFORE_RUN)
    print("=" * 80)

    if RENAME_PDFS_BEFORE_RUN:
        rename_pdfs_sequentially(PDF_DIR)

    indexed_files = build_pdf_list(START_N, END_N)

    batch_id, file_urls = apply_upload_urls(indexed_files)
    print("batch_id =", batch_id)

    upload_put(indexed_files, file_urls)

    done_map = poll_batch(
        batch_id=batch_id,
        target_names=[p.name for _, p in indexed_files],
        interval=POLL_INTERVAL,
        timeout_sec=POLL_TIMEOUT_SEC
    )

    print("\n" + "=" * 90)
    print("[done] 开始下载 / 解压 / 导出主文件到固定目录结构...")

    exports = {}
    name_to_index = {p.name: i for i, p in indexed_files}

    for file_name, item in done_map.items():
        i = name_to_index.get(file_name)
        print("\n" + "-" * 90)
        print(f"[final] {file_name} -> state={item.get('state')} -> index={i}")
        exports[file_name] = download_unzip_and_export(i, item)

    print("\n✅ 全部完成")
    print("输出根目录：", OUT_BASE)
    print("主文件导出清单：")
    print(json.dumps(exports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
