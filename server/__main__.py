"""命令行入口：python -m server 或 python run.py"""
import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())
