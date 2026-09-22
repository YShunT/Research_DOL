# 研究リポジトリ

DOL（Distribution-Aware Online Learning）の公開実装を、実行ロジックを保ちながら読みやすく再構成する研究リポジトリである。`raw/`は参照専用とし、実際の実験には`src/`を使う。

このリポジトリの `src/` は[著者公開の DOL 実装](https://github.com/cwang-nus/DOL)を参考に再構成したものです。元実装のライセンス表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に保存しています。`raw/` と処理済みデータ、学習済み checkpoint は GitHub には含めません。

[著者のデータ配布先](https://drive.google.com/drive/folders/1fpHzT_jyoHhr2uQA10-v3JTcoHCVc8IU?usp=drive_link)から Chicago-T の `chicago20_23.npz` と `adj_chicago.npy` を取得し、`datasets/chicago-t/` に配置してください。データ形式は [datasets/chicago-t/README.md](datasets/chicago-t/README.md) を参照。

## セットアップと実行

```bash
uv sync
uv run python src/run_dol.py
```

引数なしでは、既存番号の次となる`experiments/exp<ii>/`を自動作成する。先に用意した未実行の実験番号を使う場合は明示する。公開済みの`exp00`は完了済みなので、次の例では`exp01`を使う。

```bash
uv run python src/run_dol.py exp01
```

公開リポジトリには`exp00`の[レポート](experiments/exp00/report.md)と[指標](experiments/exp00/metrics.json)を含めるが、checkpointは含めない。したがって、clone直後に次の再実行コマンドは使えない。

ローカルに既存のwarm-up checkpointがある場合のみ、同じ番号のonlineを再実行できる。

```bash
uv run --offline --no-sync python src/run_dol.py exp00 --reuse-checkpoint
```

この再実行では新しいoptimizer（学習率0.001）と空のSMBを作り、validationを1回投入する。数値条件は[`src/dol/config.py`](src/dol/config.py)、実装の詳細は[`src/README.md`](src/README.md)を参照する。

## 最新の構成

```text
.
├── README.md
├── pyproject.toml              uvの依存関係とbuild設定
├── uv.lock                     解決済み依存バージョン
├── datasets/
│   └── chicago-t/              配布データの説明とローカルデータ
├── experiments/
│   ├── _template/              新規expへ複製する記録雛形
│   └── exp<ii>/                1回の実験条件・結果・考察
├── raw/                       ローカル参照用、Git公開対象外
│   └── DOL_original/          著者公開実装
├── tmp/                       一時的な検証コード、Git公開対象外
└── src/
    ├── README.md               srcの設計・rawとの対応
    ├── run_dol.py              実験入口
    ├── tests/                  継続的に使う回帰テスト
    └── dol/
        ├── artifacts.py        exp<ii>の作成と成果物保存
        ├── config.py           数値条件と保存設定
        ├── pipeline.py         warm-upからonlineまでの接続
        ├── data/               Chicago-T、標準化、窓生成
        ├── graph/              固定支持行列
        ├── layers/             LSLと拡散GCN
        ├── models/             DOL予測モデル
        ├── training/           warm-upと損失
        ├── online/             遅延教師、SMB、AH制御
        └── evaluation/         MAE、RMSE、WMAPE
```

## `experiments/exp<ii>/`との関係

|ファイル・ディレクトリ|作成者|内容|Git追跡|
|---|---|---|---|
|`strategy.md`|人|目的、前実験からの変更、仮説、成功条件|する|
|`config.json`|`src/dol/artifacts.py`|実行時の`ExperimentConfig`全体|する|
|`git_commit.txt`|`src/dol/artifacts.py`|実験開始時のcommitとdirty状態|する|
|`run.log`|`src/run_dol.py`|epoch、進捗、終了値、例外|する|
|`metrics.json`|`src/run_dol.py`|warm-up履歴、online指標、処理件数、所要時間|する|
|`checkpoints/warmup.pt`|`src/dol/training/warmup.py`|最良warm-up状態|しない|
|`arrays/*.npy`|`src/run_dol.py`|任意保存の全予測値と正解値|しない|
|`figures/`|分析コードまたは人|レポートに使う図|必要な図はする|
|`report.md`|人|結果の解釈と次の実験|する|

予測配列は大きいため既定では保存しない。必要な実験だけ`ArtifactConfig(save_predictions=True)`にする。

`src/tests/`の回帰テストはGitで管理する。一時的な検証用Pythonファイルは`tmp/`へ置き、ディレクトリごとGitから除外する（clone後は`mkdir -p tmp`で作成）。`run.log`は実験の実行記録として公開し、checkpointと大きな予測配列は除外する。

コード変更を伴う実験では、原則として変更をcommitしてから実行する。未commit変更がある場合も実行できるが、`git_commit.txt`の`dirty`が`true`になる。

