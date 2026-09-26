"""exp02: NAPL型LSLのrank別・3条件実験。"""

from __future__ import annotations

import argparse
from pathlib import Path

from dol.exp02 import RANKS, SEEDS, run_suite
from dol.evaluation.exp02_plots import plot_all, plot_distillation
from dol.evaluation.exp02_distill import run_distillation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="未完了の試行を実行する")
    parser.add_argument("--plot", action="store_true", help="保存済み結果からPNG図を生成する")
    parser.add_argument("--distill", action="store_true", help="exp01のLSL関数をrank別に近似する")
    parser.add_argument("--retry-failed", action="store_true", help="未完了試行をattemptsへ退避して再試行")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="試運転するseedだけを指定")
    parser.add_argument("--rank", type=int, choices=RANKS, help="試運転するrankだけを指定")
    arguments = parser.parse_args()
    if not arguments.run and not arguments.plot and not arguments.distill:
        parser.print_help()
        return
    if arguments.run:
        result = run_suite(
            only_seed=arguments.seed,
            only_rank=arguments.rank,
            retry_failed=arguments.retry_failed,
        )
        if result["status"] == "completed":
            plot_all()
    if arguments.plot:
        plot_all()
    if arguments.distill:
        run_distillation(only_seed=arguments.seed, only_rank=arguments.rank)
        plot_distillation(Path("experiments/exp02"))


if __name__ == "__main__":
    main()
