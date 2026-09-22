"""Chicago-Tの5 seed再現実験を実行する入口。"""

import sys

from dol.multiseed import run_suite


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_suite()
    elif sys.argv[1:] == ["--retry-failed"]:
        run_suite(retry_failed=True)
    else:
        raise SystemExit("usage: python src/run_exp01.py [--retry-failed]")
