"""TRACE 一键启动入口 — 启动前端 Streamlit 仪表盘（后端 Agent 模块内嵌加载）"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
ESSENTIAL_FILES = [
    "data/base_feature.csv",
    "data/fusion_scores.csv",
    "data/company_info.csv",
    "data/macro_quarterly.csv",
    "data/io_adjacency.csv",
    "data/graph_features.csv",
    "data/feature_importance.csv",
    "data/predictions.csv",
    "frontend/app.py",
    "backend/agent.py",
]

BANNER = r"""
  _____  _____            _____  ______
 |_   _||  __ \     /\   / ____||  ____|
   | |  | |__) |   /  \ | |     | |__
   | |  |  _  /   / /\ \| |     |  __|
   | |  | | \ \  / ____ \ |____ | |____
   |_|  |_|  \_\/_/    \_\_____||______|

  Trade-linked Risk Assessment and Contagion Engine
"""


def check_files() -> bool:
    """检查 Streamlit 仪表盘必需的数据文件是否存在。"""
    missing = [f for f in ESSENTIAL_FILES if not (PROJECT_ROOT / f).exists()]
    if missing:
        print("[!] 缺少以下必要文件：")
        for f in missing:
            print(f"   - {f}")
        print("\n[i] 请先运行数据采集和特征工程脚本 -> 参考 README.md 快速开始章节")
        return False
    return True


def check_python() -> bool:
    """检查 Python 版本。"""
    if sys.version_info < (3, 10):
        print(f"[!] Python 版本过低: {sys.version_info.major}.{sys.version_info.minor}，需要 3.10+")
        return False
    return True


def main():
    print(BANNER)
    print("  多源融合的企业信用风险智能评估与传导预警平台\n")
    print(f"  Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}  |  {sys.executable}\n")
    print("-" * 56)

    if not check_python():
        sys.exit(1)

    if not check_files():
        print("-" * 56)
        print("[!] 数据文件缺失，仅尝试启动框架（部分页面可能无数据）\n")

    print("-" * 56)
    print("[+] 正在启动 Streamlit 仪表盘...")
    print("    Local URL: http://localhost:8501\n")

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(PROJECT_ROOT / "frontend" / "app.py"),
         "--server.port", "8501"],
        cwd=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
