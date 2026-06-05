import json
import os
import re
import sys
import time
import shutil
import tempfile
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import pandas as pd
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from config_ieee import cfg


# =========================================================
# 0) 环境打印：方便写 yml
# =========================================================
def print_environment_info():
    print("=" * 80)
    print("Python 可执行文件:", sys.executable)
    print("Python 版本:", sys.version.replace("\n", " "))
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
    ]

    print(f"{'安装包名':<22}{'导入名':<22}{'版本':<18}{'状态'}")
    print("-" * 78)

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
# 1) 构造查询
# =========================================================
def build_query(cfg) -> str:
    def or_group(terms):
        return "(" + " OR ".join([f'"{t}"' if " " in t else t for t in terms]) + ")"

    topic = or_group(cfg.TOPIC_TERMS)
    control = or_group(cfg.CONTROL_TERMS)
    method = or_group(cfg.METHOD_TERMS)

    q = f"{topic} AND ({control} AND {method})"
    return q


def preview_config_and_query(cfg, query: str):
    print("\n" + "=" * 80)
    print("CONFIG PREVIEW")
    print("=" * 80)
    print("TOPIC_TERMS   =", cfg.TOPIC_TERMS)
    print("CONTROL_TERMS =", cfg.CONTROL_TERMS)
    print("METHOD_TERMS  =", cfg.METHOD_TERMS)
    print("YEAR RANGE    =", cfg.START_YEAR, "to", cfg.END_YEAR)
    print("MAX_ITEMS     =", cfg.MAX_ITEMS)
    print("ROWS_PER_PAGE =", cfg.ROWS_PER_PAGE)
    print("SLEEP_S       =", cfg.SLEEP_S)
    print("SORT_BY       =", cfg.SORT_BY)
    print("SORT_DESC     =", cfg.SORT_DESC)
    print("TOP_N_AFTER_SORT =", cfg.TOP_N_AFTER_SORT)
    print("DOWNLOAD_PDFS =", cfg.DOWNLOAD_PDFS)
    print("\nQUERY:\n", query)


# =========================================================
# 2) 浏览器启动（每次运行使用全新临时 profile）
# =========================================================
def ensure_dirs(cfg):
    cfg.TEMP_PROFILE_PARENT.mkdir(parents=True, exist_ok=True)
    cfg.BASE_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PDF_DIR.mkdir(parents=True, exist_ok=True)


def start_driver_fresh(headless=False, temp_profile_parent: Path = None):
    """
    每次运行创建一个全新的临时 Chrome profile，
    从而强制用户重新登录，不复用历史 cookie。
    本次运行结束后该 profile 会被删除。
    """
    if temp_profile_parent is None:
        temp_profile_parent = Path("./tmp_profiles").resolve()

    temp_profile_parent.mkdir(parents=True, exist_ok=True)

    temp_profile_dir = Path(
        tempfile.mkdtemp(prefix="ieee_profile_", dir=str(temp_profile_parent))
    ).resolve()

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")

    # 每次运行都用新的临时 profile
    opts.add_argument(f"--user-data-dir={temp_profile_dir}")

    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--profile-directory=Default")

    prefs = {
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "plugins.always_open_pdf_externally": True,
    }
    opts.add_experimental_option("prefs", prefs)
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=opts)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": (
                "Object.defineProperty(navigator, 'webdriver', "
                "{get: () => undefined})"
            )
        }
    )
    return driver, temp_profile_dir


# =========================================================
# 3) Session 构造
# =========================================================
def build_session_from_driver(driver) -> requests.Session:
    s = requests.Session()

    for c in driver.get_cookies():
        s.cookies.set(c.get("name"), c.get("value"))

    try:
        ua = driver.execute_script("return navigator.userAgent;")
        s.headers.update({"User-Agent": ua})
    except Exception:
        pass

    return s


def requests_session_from_driver(driver) -> requests.Session:
    s = requests.Session()

    try:
        ua = driver.execute_script("return navigator.userAgent;")
        s.headers.update({"User-Agent": ua})
    except Exception:
        pass

    for c in driver.get_cookies():
        s.cookies.set(c["name"], c["value"])

    return s


# =========================================================
# 3.5) 同一次运行内：等待用户重新登录
# =========================================================
def wait_for_user_login(driver, reason: str = ""):
    print("\n" + "=" * 80)
    print("需要等待你在当前浏览器中完成 IEEE 登录")
    if reason:
        print("原因：", reason)
    print("请在刚打开的 Chrome 窗口中完成 IEEE / 机构登录。")
    print("不要关闭这个浏览器窗口。")
    print("登录完成后，回到终端按回车，脚本会在本次运行内继续。")
    print("=" * 80)

    try:
        driver.get(cfg.IEEE_HOME_URL)
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ 打开 IEEE 首页失败，但仍可手动在当前浏览器中完成登录: {e}")

    input(">>> 登录完成后按回车继续 ... ")
    time.sleep(2)


# =========================================================
# 4) IEEE 查询修正
# =========================================================
def sanitize_ieee_query(q: str) -> str:
    if not isinstance(q, str):
        q = str(q)

    q = q.replace("RRT*", "RRT star")
    q = q.replace("A*", "A star")
    q = q.replace("D* Lite", "D star lite").replace("D* lite", "D star lite")
    q = q.replace("D*", "D star")

    q = re.sub(r"\b([A-Za-z]{1,2})\*\b", r"\1 star", q)
    return q


# =========================================================
# 5) IEEE REST Search（加入自动重试）
# =========================================================
def ieee_rest_search(session, query, rows=25, page=1, max_retries=6, backoff_base=2.0):
    url = "https://ieeexplore.ieee.org/rest/search"

    query_clean = sanitize_ieee_query(str(query))

    payload = {
        "newsearch": True,
        "queryText": query_clean,
        "highlight": True,
        "returnFacets": ["ALL"],
        "returnType": "SEARCH",
        "rowsPerPage": int(rows),
        "pageNumber": int(page),
    }

    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json;charset=UTF-8",
        "Origin": "https://ieeexplore.ieee.org",
        "Referer": cfg.IEEE_SEARCH_REFERER,
        "X-Requested-With": "XMLHttpRequest",
    }

    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            r = session.post(url, headers=headers, json=payload, timeout=30)

            if r.status_code == 200:
                return r.json()

            transient_codes = {429, 500, 502, 503, 504}

            print(
                f"❌ REST search failed | page={page} | "
                f"attempt={attempt}/{max_retries} | status={r.status_code}"
            )
            print("---- response head (800 chars) ----")
            print((r.text or "")[:800])

            if r.status_code in transient_codes and attempt < max_retries:
                sleep_s = backoff_base * attempt
                print(f"⏳ IEEE 临时不可用，等待 {sleep_s:.1f}s 后重试 page {page} ...")
                time.sleep(sleep_s)
                continue

            print("---- query(before) ----")
            print(str(query)[:1000])
            print("---- query(after sanitize) ----")
            print(query_clean[:1000])
            print("---- payload ----")
            print(payload)
            r.raise_for_status()

        except requests.exceptions.RequestException as e:
            last_err = e
            print(
                f"❌ REST search exception | page={page} | "
                f"attempt={attempt}/{max_retries} | error={e}"
            )

            if attempt < max_retries:
                sleep_s = backoff_base * attempt
                print(f"⏳ 等待 {sleep_s:.1f}s 后重试 page {page} ...")
                time.sleep(sleep_s)
                continue

            raise

    if last_err is not None:
        raise last_err

    raise RuntimeError(f"REST search failed unexpectedly at page={page}")


# =========================================================
# 6) 数据清洗
# =========================================================
def join_authors(authors_field):
    if isinstance(authors_field, dict):
        authors_field = authors_field.get("authors", [])

    if not isinstance(authors_field, list):
        return ""

    names = []
    for a in authors_field:
        if isinstance(a, dict):
            names.append(
                a.get("preferredName")
                or a.get("fullName")
                or a.get("name")
                or ""
            )

    return ", ".join([n for n in names if n])


def normalize_record(rec: dict):
    arnumber = str(rec.get("articleNumber") or rec.get("arnumber") or "").strip()
    title = (rec.get("articleTitle") or "").strip()
    year = rec.get("publicationYear") or ""
    pub_date = rec.get("publicationDate") or ""
    doi = (rec.get("doi") or "").strip()
    authors = join_authors(rec.get("authors", []))

    citation_count = rec.get("citationCount", 0)
    download_count = rec.get("downloadCount", 0)
    is_oa = bool(rec.get("isOpenAccess") or rec.get("openAccess") or False)

    doc_link = rec.get("documentLink") or (f"/document/{arnumber}" if arnumber else "")
    if doc_link and doc_link.startswith("/"):
        doc_link = "https://ieeexplore.ieee.org" + doc_link
    if doc_link and not doc_link.endswith("/"):
        doc_link += "/"

    pdf_link = rec.get("pdfLink") or ""
    if pdf_link and pdf_link.startswith("/"):
        pdf_link = "https://ieeexplore.ieee.org" + pdf_link

    try:
        citation_count = int(citation_count)
    except Exception:
        pass

    try:
        download_count = int(download_count)
    except Exception:
        pass

    return {
        "arnumber": arnumber,
        "title": title,
        "year": year,
        "publicationDate": pub_date,
        "authors": authors,
        "doi": doi,
        "is_open_access": is_oa,
        "citationCount": citation_count,
        "downloadCount": download_count,
        "publicationTitle": rec.get("publicationTitle") or "",
        "contentType": rec.get("contentType") or rec.get("articleContentType") or "",
        "document_url": doc_link,
        "pdfLink": pdf_link,
        "pdfSize": rec.get("pdfSize") or "",
        "citationsLink": rec.get("citationsLink") or "",
    }


# =========================================================
# 7) 分页抓取 + 年份过滤（加入容错）
# =========================================================
def fetch_filtered_n(
    session,
    query,
    target_n=100,
    rows_per_page=25,
    sleep_s=0.8,
    start_year=2019,
    end_year=2025,
):
    """
    最终返回 target_n 条（过滤后也要凑够）
    对 IEEE REST 临时 502/503/504/429 自动重试；
    如果某页最终仍失败，则保留当前已抓到的数据并退出，不让整个程序直接崩掉。
    """
    collected = []
    page = 1
    total = None
    seen = set()

    while len(collected) < target_n:
        try:
            data = ieee_rest_search(
                session=session,
                query=query,
                rows=rows_per_page,
                page=page,
                max_retries=6,
                backoff_base=2.0,
            )
        except Exception as e:
            print(f"\n⚠️ page {page} 连续重试后仍失败：{e}")
            print("⚠️ 不再让程序整体崩掉，保留当前已抓到的数据并提前结束检索。")
            break

        if total is None:
            total = data.get("totalRecords")
            print("✅ totalRecords:", total)

        recs = data.get("records", [])
        print(f"page {page}: returned {len(recs)}")

        if not recs:
            break

        for rec in recs:
            row = normalize_record(rec)
            ar = row.get("arnumber", "")
            if not ar or ar in seen:
                continue
            seen.add(ar)

            y = row.get("year", "")
            try:
                y = int(y)
            except Exception:
                y = None

            if y is None or (start_year <= y <= end_year):
                collected.append(row)
                if len(collected) >= target_n:
                    break

        page += 1
        time.sleep(sleep_s)

        if total and page > (total // rows_per_page + 5):
            break

    print(f"\n✅ 检索结束：共收集 {len(collected)} 条过滤后记录")
    return pd.DataFrame(collected)


# =========================================================
# 8) 导出 metadata
# =========================================================
def save_metadata(df100: pd.DataFrame, params: dict, cfg):
    df100.to_csv(cfg.META_CSV, index=False, encoding="utf-8-sig")
    with open(cfg.PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print("✅ saved:", str(cfg.META_CSV))
    print("✅ saved:", str(cfg.PARAMS_JSON))


# =========================================================
# 9) venue 简易识别
# =========================================================
def venue_from_doi(doi: str):
    if not isinstance(doi, str):
        return ("unknown", "unknown")

    d = doi.upper()

    if "IROS" in d:
        return ("conf", "IROS")
    if "ICRA" in d:
        return ("conf", "ICRA")
    if "HUMANOIDS" in d:
        return ("conf", "Humanoids")
    if "CVPR" in d:
        return ("conf", "CVPR")
    if "ICASSP" in d:
        return ("conf", "ICASSP")
    if "SII" in d:
        return ("conf", "SII")
    if "OCEANS" in d:
        return ("conf", "OCEANS")

    if "TRO" in d:
        return ("journal", "TRO")
    if "LRA" in d:
        return ("journal", "RA-L")
    if "TNNLS" in d:
        return ("journal", "TNNLS")
    if "TITS" in d:
        return ("journal", "TITS")
    if "TIE" in d:
        return ("journal", "TIE")
    if "TII" in d:
        return ("journal", "TII")
    if "TAI" in d:
        return ("journal", "TAI")
    if "TASE" in d:
        return ("journal", "TASE")
    if "TVT" in d:
        return ("journal", "TVT")
    if "ACCESS" in d:
        return ("journal", "IEEE Access")

    return ("unknown", "unknown")


# =========================================================
# 10) PDF 下载
# =========================================================
def sanitize_filename(s: str, maxlen=120):
    s = re.sub(r'[\\/:*?"<>|]+', "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:maxlen]


def stamp_to_getpdf(stamp_url: str) -> str:
    qs = parse_qs(urlparse(stamp_url).query)
    ar = (qs.get("arnumber") or [""])[0]
    if not ar:
        raise ValueError(f"stamp_url 里没 arnumber: {stamp_url}")
    return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={ar}&ref="


def download_pdf_via_requests(driver, stamp_url: str, arnumber: str, title: str, out_dir: Path):
    s = requests_session_from_driver(driver)
    pdf_url = stamp_to_getpdf(stamp_url)

    fname = f"{arnumber}_{sanitize_filename(title)}.pdf"
    out = out_dir / fname
    part = out_dir / f"{fname}.part"

    if part.exists():
        try:
            part.unlink()
        except Exception:
            pass

    try:
        r = s.get(pdf_url, stream=True, timeout=60, allow_redirects=True)
        ct = (r.headers.get("Content-Type") or "").lower()

        if r.status_code != 200 or "application/pdf" not in ct:
            head = r.text[:400] if "text" in ct or ct == "" else str(r.headers)
            raise RuntimeError(
                f"没拿到PDF: status={r.status_code}, content-type={ct}\n{head}"
            )

        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

        if (not part.exists()) or part.stat().st_size == 0:
            raise RuntimeError("下载结果为空文件，视为失败")

        part.replace(out)
        return str(out)

    except Exception:
        if part.exists():
            try:
                part.unlink()
            except Exception:
                pass
        raise


def download_pdfs_from_csv(driver, csv_path: Path, out_dir: Path):
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    n = len(df)

    if n == 0:
        print("⚠️ metadata 为空，跳过 PDF 下载。")
        return

    print("\n" + "=" * 80)
    print("即将开始 PDF 下载。")
    print("脚本会先等待你在当前 Chrome 中完成一次 IEEE 登录。")
    print("如果中途某篇下载失败，脚本会在本次运行内再次等待你登录，然后重试当前论文。")
    print("=" * 80)

    wait_for_user_login(
        driver,
        reason="开始 PDF 下载前，先确认登录状态。"
    )

    ok = 0
    skipped = 0
    failed = 0

    for i in range(n):
        row = df.iloc[i]
        stamp_url = row["pdfLink"]
        arnumber = str(row["arnumber"])
        title = str(row["title"])

        target = list(out_dir.glob(f"{arnumber}_*.pdf"))
        if target:
            skipped += 1
            print(f"⏭️  已存在，跳过 {arnumber}")
            continue

        max_attempts = 2
        attempt = 1

        while attempt <= max_attempts:
            try:
                path = download_pdf_via_requests(driver, stamp_url, arnumber, title, out_dir)
                ok += 1
                print(f"✅ 下载成功 {arnumber} -> {path}")
                time.sleep(1.0)
                break

            except Exception as e:
                print(f"❌ 下载失败 {arnumber} | 第 {attempt}/{max_attempts} 次尝试")
                print("   错误信息：", e)

                if attempt >= max_attempts:
                    failed += 1
                    print(f"⚠️  {arnumber} 已在本次运行内重新登录后重试过，仍失败，跳过。")
                    break

                wait_for_user_login(
                    driver,
                    reason=f"论文 {arnumber} 下载失败。请重新登录后继续，脚本将自动重试当前论文。"
                )
                attempt += 1

    print("\n" + "=" * 80)
    print(f"完成：成功 {ok} 篇，已存在跳过 {skipped} 篇，失败 {failed} 篇")
    print(f"输出目录：{out_dir.resolve()}")
    print("=" * 80)


# =========================================================
# 11) 主流程
# =========================================================
def main():
    ensure_dirs(cfg)

    print_environment_info()

    query = build_query(cfg)
    preview_config_and_query(cfg, query)

    params = {
        "query_text": query,
        "max_items": cfg.MAX_ITEMS,
        "rows_per_page": cfg.ROWS_PER_PAGE,
        "sleep_s": cfg.SLEEP_S,
        "start_year": cfg.START_YEAR,
        "end_year": cfg.END_YEAR,
        "sort_by": cfg.SORT_BY,
        "sort_desc": cfg.SORT_DESC,
        "top_n_after_sort": cfg.TOP_N_AFTER_SORT,
    }

    driver = None
    temp_profile_dir = None

    try:
        driver, temp_profile_dir = start_driver_fresh(
            headless=cfg.HEADLESS,
            temp_profile_parent=cfg.TEMP_PROFILE_PARENT
        )
        driver.get(cfg.IEEE_HOME_URL)
        time.sleep(3)

        print("\n✅ Chrome 已用【临时 profile】启动：", temp_profile_dir)
        print("👉 本次运行里，脚本会在需要时等待你登录；不用退出脚本再重跑。")
        print("👉 脚本结束后，该临时登录态不会保留。")

        session = build_session_from_driver(driver)
        print("✅ session ready, cookies:", len(driver.get_cookies()))

        df100 = fetch_filtered_n(
            session=session,
            query=params["query_text"],
            target_n=params["max_items"],
            rows_per_page=params["rows_per_page"],
            sleep_s=params["sleep_s"],
            start_year=params["start_year"],
            end_year=params["end_year"],
        )

        sort_key = params.get("sort_by", "citationCount")
        if sort_key in df100.columns:
            df100 = df100.sort_values(
                sort_key,
                ascending=not params.get("sort_desc", True)
            ).reset_index(drop=True)

        show_cols = [
            "arnumber", "title", "year", "publicationDate", "authors",
            "is_open_access", "citationCount", "downloadCount",
            "document_url", "pdfLink", "doi"
        ]
        show_cols = [c for c in show_cols if c in df100.columns]

        df100 = df100[show_cols].head(cfg.TOP_N_AFTER_SORT).reset_index(drop=True)

        print("\n✅ df100 rows:", len(df100))
        print(df100.head(cfg.TOP_N_AFTER_SORT).to_string(max_colwidth=80))

        save_metadata(df100, params, cfg)

        df100[["venue_type", "venue"]] = df100["doi"].apply(
            lambda x: pd.Series(venue_from_doi(x))
        )

        # 这里如果你想把 venue_type / venue 也写回 csv，可以取消下一行注释
        # df100.to_csv(cfg.META_CSV, index=False, encoding="utf-8-sig")

        if cfg.DOWNLOAD_PDFS:
            download_pdfs_from_csv(driver, cfg.META_CSV, cfg.PDF_DIR)

    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass

        if temp_profile_dir is not None and Path(temp_profile_dir).exists():
            try:
                shutil.rmtree(temp_profile_dir, ignore_errors=True)
                print(f"🧹 已删除临时 profile: {temp_profile_dir}")
            except Exception as e:
                print(f"⚠️ 删除临时 profile 失败: {temp_profile_dir}, error={e}")


if __name__ == "__main__":
    main()