from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # =========================
    # 1) 检索词配置
    # =========================
    TOPIC_TERMS: list = field(default_factory=lambda: [
        "actor-critic reinforcement learning",
        "actor critic",
        "actor-critic methods",
        "policy gradient",
        "value function approximation",
        "advantage actor-critic",
        "A2C",
        "A3C",
        "deep reinforcement learning",
    ])

    CONTROL_TERMS: list = field(default_factory=lambda: [
        "policy optimization",
        "value estimation",
        "advantage estimation",
        "exploration and exploitation",
        "continuous control",
        "discrete control",
        "reward shaping",
        "sample efficiency",
        "stability and convergence",
        "off-policy learning",
    ])

    METHOD_TERMS: list = field(default_factory=lambda: [
        "proximal policy optimization",
        "PPO",
        "soft actor-critic",
        "SAC",
        "deep deterministic policy gradient",
        "DDPG",
        "twin delayed deep deterministic policy gradient",
        "TD3",
        "generalized advantage estimation",
        "GAE",
    ])
    # =========================
    # 2) 检索参数
    # =========================
    START_YEAR: int = 2000
    END_YEAR: int = 2026
    MAX_ITEMS: int = 100
    ROWS_PER_PAGE: int = 25
    SLEEP_S: float = 0.8

    SORT_BY: str = "citationCount"   # citationCount / downloadCount / publicationDate
    SORT_DESC: bool = True

    # ⚠️ 这个很重要：排序后只保留前100
    TOP_N_AFTER_SORT: int = 10

    # =========================
    # 3) 浏览器配置
    # =========================
    HEADLESS: bool = False

    # 不再使用固定 profile；改成临时 profile 的父目录
    TEMP_PROFILE_PARENT: Path = Path("./tmp_profiles").resolve()

    # =========================
    # 4) 输出目录配置
    # =========================
    BASE_DIR: Path = Path("./Paper_test").resolve()
    PDF_DIR: Path = Path("./Paper_test/pdfs").resolve()
    META_CSV: Path = Path("./Paper_test/ieee_top100_metadata.csv").resolve()
    PARAMS_JSON: Path = Path("./Paper_test/ieee_query_params.json").resolve()

    # =========================
    # 5) 执行开关
    # =========================
    DOWNLOAD_PDFS: bool = True

    # =========================
    # 6) 其他
    # =========================
    IEEE_HOME_URL: str = "https://ieeexplore.ieee.org/"
    IEEE_SEARCH_REFERER: str = "https://ieeexplore.ieee.org/search/searchresult.jsp"


cfg = Config()
