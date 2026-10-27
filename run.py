#!/usr/bin/env python
"""启动小智本地控制中心。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server.app import main

if __name__ == "__main__":
    main()
