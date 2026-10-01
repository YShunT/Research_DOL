# 研究リポジトリ

DOL（Distribution-Aware Online Learning for Urban Spatiotemporal Forecasting on
Streaming Data）の公開実装を、実行ロジックを保ちながら読みやすく再構成する研究リポジトリである。`raw/`は参照専用とし、実際の実験には`src/`を使う。

このリポジトリの `src/` は[著者公開の DOL 実装](https://github.com/cwang-nus/DOL)を参考に再構成したものです。元実装のライセンス表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に保存しています。`raw/` と処理済みデータ、学習済み checkpoint は GitHub には含めません。

## 実験の実行記録

|実行日（JST）|実験番号|実験内容|
|---|---|---|
|2026-09-21|[exp00](experiments/exp00/report.md)|Chicago-Tでseed 42の論文再現実験。warm-up checkpointを使ったonline評価を実施。|
|2026-09-22|[exp01](experiments/exp01/report.md)|Chicago-Tでseed 42〜46の5試行を、それぞれwarm-upから実行し、論文値と平均・標準偏差を比較。|
|2026-09-23|[exp02](experiments/exp02/report.md)|地点別LSLの重みを`Θ=1θ_sharedᵀ+EB`または`Θ=EB`で生成するNAPL型低ランクLSLを、rank 0〜64・5 seedでwarm-upから学習して比較。`Θ=EB`をonline更新する条件はrank 4（1,476パラメータ）でexp01比1%以内に収まった。|
|2026-09-29|[exp03](experiments/exp03/report.md)|exp01の学習済みLSLを再学習なしで層別SVD近似し、重み・補正関数・予測・online軌跡の冗長性を検証。重みに地点間の低rank構造はなく、元より少ない保持数では補正関数を保てなかった。補正の大半は地点別biasが担っていた。|

## 最新の構成

```text
.
├── README.md
├── pyproject.toml              uvの依存関係とbuild設定
├── uv.lock                     解決済み依存バージョン
├── datasets/
│   └── chicago-t/              配布データの説明とローカルデータ
├── experiments/
│   ├── _template/              任意のローカル雛形（Git対象外）
│   └── exp<ii>/                実験条件・結果・考察
├── raw/                       ローカル参照用、Git公開対象外
│   └── DOL_original/          著者公開実装
├── tmp/                       一時的な検証コード、Git公開対象外
└── src/
    ├── README.md               srcの設計・rawとの対応
    ├── run_dol.py              単発実験入口
    ├── run_exp01.py            5 seed実験入口
    ├── run_exp02.py            低ランクLSLのrank比較の入口
    ├── run_exp03.py            層別SVDによる冗長性検証の入口
    └── dol/
        ├── artifacts.py        exp<ii>の作成と成果物保存
        ├── multiseed.py        exp01の順次実行・集計・再試行
        ├── exp02.py            exp02のrank×条件×seedの実行管理・集計
        ├── exp03/              exp03のSVD、関数診断、介入評価、軌跡解析、作図
        ├── config.py           数値条件と保存設定
        ├── pipeline.py         warm-upからonlineまでの接続
        ├── data/               Chicago-T、標準化、窓生成
        ├── graph/              固定支持行列
        ├── layers/             LSL、NAPL型低ランクLSL、拡散GCN
        ├── models/             DOL予測モデル
        ├── training/           warm-upと損失
        ├── online/             遅延教師、SMB、AH制御
        └── evaluation/         MAE、global/sample-wise RMSE、WMAPE、exp02の評価・作図
```

実験の計画・設定・ログ・指標・レポートをGitで共有する。`raw/`、処理済みデータ、checkpoint、大きな予測配列、ローカルの実験雛形は公開対象外とする。
