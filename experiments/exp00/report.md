# exp00 実験レポート：Chicago-T における DOL 再現

## 目的と条件

著者配布の Chicago-T データで、DOL のオンライン予測性能を[論文 Table 2](https://www.ijcai.org/proceedings/2025/0372.pdf)と比較する。入力・予測は各12ステップ、train/validation/online は時系列順に 20%/5%/75%、77地点、seed 42。warm-up は AdamW、最大150 epoch・patience 10 で行い、最良の epoch 25 の checkpoint を使用した。今回の再実行では checkpoint を保持し、オンライン用に新しい optimizer（学習率0.001、空の状態）と SMB（容量1,000）を構築した。validation 7,002窓を SMB に1回投入し、1週間 Awake・1週間 Hibernate、EM 8件でオンライン適応した。詳しい設定は [config.json](config.json) と [strategy.md](strategy.md) を参照。

## 結果

`metrics.json` は `completed`。2026-09-21 17:52:48〜18:29:09 JST にオンライン 105,181窓（97,187,244値）を処理した。52,765ステップで更新し、Awake 52,765・Hibernate 52,416ステップだった。全実行時間は2,181秒、オンライン評価は2,169秒。checkpoint の SHA-256 は `1e280e6d3512cb88734d255c4d3a6d9e8c29db3802098ecef806eb78d4b707f1` で、再実行前後で不変。

|指標|exp00（seed 42）|論文 Chicago-T DOL（5 seed の平均±標準偏差）|差（exp00−論文平均）|
|---|---:|---:|---:|
|MAE|0.698637|0.72±0.00|−0.021363（−2.97%）|
|global RMSE|2.215612|2.06±0.02|+0.155612（+7.55%）|
|WMAPE|35.8900%|36.80%±0.19%|−0.9100ポイント（−2.47%）|

補助指標の sample-wise RMSE は 1.987183。これは各窓で RMSE を計算してから窓平均した値であり、global RMSE とは異なる。論文との比較には global RMSE を用いた。上表の論文値と5回反復の条件は[論文 Table 2](https://www.ijcai.org/proceedings/2025/0372.pdf)による。

## 解釈と限界

- 「論文より良い」のは MAE と WMAPE のみで、global RMSE は論文値より 7.55% 高い。3指標すべてで上回ったわけではなく、strategy.md の暫定成功条件（3指標とも相対差5%以内）は満たさない。
- この3指標は既存の公開実装 `raw/` を seed 42 で実行した結果（MAE 0.698637、RMSE 2.215612、WMAPE 35.8900%）と一致する。再実装の先頭100窓の予測も raw と完全一致した。したがって、少なくとも観測された差を `src/` 独自の指標計算やオンライン初期化だけで説明する根拠はない。
- 論文値は5 seed の集計だが、exp00 は seed 42 の1試行。丸められた論文平均・標準偏差から seed 42 の原値は分からない。とくに RMSE の差は報告標準偏差0.02より大きく、単に「論文を上回った」とは評価できない。著者が用いた checkpoint・seed・データ版・評価実装の差は未確定である。
- `sample_rmse` は global RMSE より低いが、これを論文の RMSE 2.06 と同一指標として比較しない。`raw/` の `utils.metrics.RMSE` も全予測値を連結した global RMSE を計算する。
- この実行では全予測配列を保存していないため、時期別・地域別の誤差や大誤差事例の追加分析はこの成果物だけではできない。再現性の監査時には `git_commit.txt` が `unborn` / `dirty: true` である点にも注意が必要。
- 前回の試行は再実行時に `attempts/` へ退避された後、ユーザーの指定で削除した。`metrics.json` の `previous_attempt` は削除済みパスを示す履歴情報であり、旧ログ・旧指標は残っていない。

## 次の実験

1. seed を変えた計5試行で平均・標準偏差を求め、論文と同じ集計単位で比較する。
2. RMSE 差を調べるため、必要なら予測配列を保存して時期別・地点別・需要ピーク別の二乗誤差を分析する。
