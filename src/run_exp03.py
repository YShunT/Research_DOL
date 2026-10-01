"""exp03: 層別SVDで既存LSLの冗長性を調べる（主実験A〜C）。"""
from __future__ import annotations

import argparse
from pathlib import Path

from dol.exp03.runner import SEEDS, Settings, run_suite
from dol.exp03.plots import plot_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="主実験A〜Cを実行・再開")
    parser.add_argument("--plot", action="store_true", help="保存済みJSONからPNGを生成")
    parser.add_argument("--stage", choices=("all", "spectra", "static", "trajectory"), default="all",
                        help="spectra=A, static=A+B, trajectory=C")
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32, help="更新なし推論のbatch size")
    parser.add_argument("--diagnostic-count", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=1, help="CPU PyTorch threads")
    parser.add_argument("--smoke", action="store_true", help="8窓・少数rankの接続確認")
    parser.add_argument("--output", type=Path, help="既定: experiments/exp03; smoke: tmp/exp03-smoke")
    parser.add_argument("--retry-failed", action="store_true", help="失敗したonline軌跡を退避して再実行")
    arguments = parser.parse_args()
    if not arguments.run and not arguments.plot:
        parser.print_help()
        return
    root = arguments.output or Path("tmp/exp03-smoke" if arguments.smoke else "experiments/exp03")
    if arguments.run:
        settings = Settings(arguments.device, arguments.batch_size, arguments.diagnostic_count,
                            arguments.smoke, arguments.threads)
        run_suite(root, settings, stage=arguments.stage, only_seed=arguments.seed,
                  retry_failed=arguments.retry_failed)
        plot_all(root)
    elif arguments.plot:
        plot_all(root)


if __name__ == "__main__":
    main()
