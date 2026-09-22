# exp01 実験レポート：Chicago-Tの5 seed再現実験

## 目的と条件

[実行前計画](strategy.md)に従い、著者配布のChicago-TデータでDOLをseed 42、43、44、45、46の5回、warm-upから独立に実行した。[DOL論文 Table 2](https://www.ijcai.org/proceedings/2025/0372.pdf)のChicago-Tにおける5試行の報告値と、online期間全体のMAE、global RMSE、WMAPEを比較する。sample-wise RMSEは補助指標として別に集計し、論文のRMSEの代用にはしない。

時系列分割はtrain/validation/onlineが20%/5%/75%、入力・予測長は各12ステップ、地点数は77。各seedで最良validation checkpointのモデル重みとscalerを使用し、online用のoptimizerとSMBは新規に構築した。online開始時の学習率は0.001、optimizer stateとSMBは空で、観測済みのvalidation 7,002窓をSMBへ1回投入した。設定と入力ファイルのSHA-256は[config.json](config.json)、seed別の詳細は`seeds/seed<番号>/config.json`と`metrics.json`を参照。

## 実行結果

2026-09-22 13:24:06～19:19:33 JSTに5試行すべて`completed`となった。各試行はonline 105,181窓を処理し、52,765回更新した（Awake 52,765ステップ、Hibernate 52,416ステップ）。5試行とも`run_mode=warmup_then_online`で、checkpointが存在し、記録したSHA-256と一致した。実行開始時のコードは[git_commit.txt](git_commit.txt)記載の`eeae0c53d406d4f7162eadeabc7a5513d29d5135`で、`dirty: false`だった。[run.log](run.log)に試行の開始・終了、各seedの`run.log`にwarm-upとonlineの進捗が残る。

|seed|最良warm-up epoch|MAE|global RMSE|sample-wise RMSE|WMAPE|
|---:|---:|---:|---:|---:|---:|
|42|25|0.698637|2.215612|1.987183|35.8900%|
|43|8|0.706143|2.267364|2.025183|36.2756%|
|44|26|0.699343|2.225922|1.995416|35.9263%|
|45|5|0.706964|2.264251|2.025531|36.3178%|
|46|11|0.704040|2.246196|2.011656|36.1675%|

下表のexp01は5 seedの算術平均±標本標準偏差（`ddof=1`）。差率は`(exp01平均−論文平均)/論文平均`で計算した。WMAPEは`metrics.json`の比率を100倍して表示する。

|指標|exp01 平均±標準偏差|論文 Chicago-T DOL|論文平均との差率|事前基準：平均が相対5%以内|
|---|---:|---:|---:|---|
|MAE|0.703025±0.003843|0.72±0.00|−2.36%|満たす|
|global RMSE|2.243869±0.022877|2.06±0.02|+8.93%|満たさない|
|WMAPE|36.1154%±0.1974ポイント|36.80%±0.19ポイント|−1.86%|満たす|
|sample-wise RMSE|2.008994±0.017340|-|-|-|

各指標の元値と集計値は[metrics.json](metrics.json)にある。WMAPEの平均は論文より0.6846ポイント低い。

## 解釈と限界

- 5試行すべてが同一コード版・入力データ・固定条件で完了し、seed以外の条件を揃えるという実行面の目的は達成した。MAEとWMAPEの平均は事前に定めた相対5%以内だが、global RMSEは8.93%高い。したがって、3主指標すべてが5%以内という**事前の数値再現基準は満たさない**。
- RMSEは5 seedとも論文平均2.06を上回り、範囲は2.215612～2.267364だった。exp00のseed 42だけに由来する上振れではない。ただし論文はseed番号・丸め前の個別値を示していないため、原因をseed差だけで否定したり、実装・データ・評価方法のどれかに特定したりはできない。論文の標準偏差も表示桁で丸められている。
- seed42のMAE、global RMSE、sample-wise RMSE、WMAPEは[exp00](../exp00/report.md)と数値が完全一致した。独立に作ったwarm-up checkpointのモデル・scalerのテンソルも一致したが、checkpointファイルのSHA-256は異なるため、ファイル自体のバイト一致とは区別する。
- sample-wise RMSEの平均は2.008994で、global RMSEの平均2.243869より低い。窓ごとに平方根を取ってから平均するか、全窓をまとめて平方根を取るかの違いであり、前者を論文値2.06と比較して再現成功とは判定しない。
- この実験では全予測配列を保存していない。需要のピーク、地点、時期ごとの二乗誤差をこの成果物だけから再集計できないため、RMSE差の発生箇所や理由は未確定である。GPU等の実行環境も論文のRTX 3090と同一条件ではなく、所要時間を論文の速度再現とは見なさない。
