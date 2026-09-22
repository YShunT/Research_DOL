# 研究リポジトリ

DOL（Distribution-Aware Online Learning for Urban Spatiotemporal Forecasting on
Streaming Data）の公開実装を、実行ロジックを保ちながら読みやすく再構成する研究リポジトリである。`raw/`は参照専用とし、実際の実験には`src/`を使う。

このリポジトリの `src/` は[著者公開の DOL 実装](https://github.com/cwang-nus/DOL)を参考に再構成したものです。元実装のライセンス表示は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) に保存しています。`raw/` と処理済みデータ、学習済み checkpoint は GitHub には含めません。


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
    └── dol/
        ├── artifacts.py        exp<ii>の作成と成果物保存
        ├── multiseed.py        exp01の順次実行・集計・再試行
        ├── config.py           数値条件と保存設定
        ├── pipeline.py         warm-upからonlineまでの接続
        ├── data/               Chicago-T、標準化、窓生成
        ├── graph/              固定支持行列
        ├── layers/             LSLと拡散GCN
        ├── models/             DOL予測モデル
        ├── training/           warm-upと損失
        ├── online/             遅延教師、SMB、AH制御
        └── evaluation/         MAE、global/sample-wise RMSE、WMAPE
```

実験の計画・設定・ログ・指標・レポートをGitで共有する。`raw/`、処理済みデータ、checkpoint、大きな予測配列、ローカルの実験雛形は公開対象外とする。
