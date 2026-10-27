"""小智本地控制中心服务端。"""
import sys
from pathlib import Path

__version__ = "0.1.0"

# --- 工作区内置依赖引导 -------------------------------------------------
# 线上服务跑在只读根文件系统的 /usr/bin/python3 上，site-packages 里没有
# httpx / websockets（pip 也装不进去：PEP 668 + 根目录只读）。
# 这里把随项目发布的 _pylibs/ 挂到 sys.path 末尾：
#   - 缺失的包（httpx、websockets、httpcore、h11）从这里补齐；
#   - 系统已有的包（anyio、certifi、idna）仍优先用系统版本，不会被覆盖。
_VENDOR = Path(__file__).resolve().parent.parent / "_pylibs"
if _VENDOR.is_dir():
    _vendor_path = str(_VENDOR)
    if _vendor_path not in sys.path:
        sys.path.append(_vendor_path)
