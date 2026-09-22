# exp00：DOL論文のChicago-T再現実験

## 実験目的

本実験の主目的は、DOL（Distribution-Aware Online Learning）論文の表2で報告されたChicago-Tの結果を、配布済みデータと本リポジトリの再実装で再現できるか検証することである。具体的には、論文と同じ時系列分割、Graph WaveNetベースの予測器、地点固有学習器（LSL）、Streaming Memory Buffer（SMB）、Awake/Hibernate（AH）スケジュールを用い、オンライン期間全体のMAE、RMSE、WMAPEを比較する。

あわせて、次の実装上の再現性を確認する。

- warm-upからonline評価までが一貫して実行できること。
- online評価で未来の正解を先取りせず、12ステップ遅延した教師だけを更新に使うこと。
- Awake中はLSLのみを更新し、Hibernate中は予測のみを行うこと。
- SMBのリザーバサンプリングと休眠開始時のリセットが、公開実装と同じ条件で動作すること。
- rawと同じく全予測値をCPUに保持し、論文と比較可能なglobal RMSEを一括集計できること。

論文のChicago-TにおけるDOL報告値を比較基準とする。

|指標|論文報告値|
|---|---:|
|MAE|0.72|
|RMSE|2.06|
|WMAPE|36.80%|

本実験はDOL単体の再現を対象とする。13ベースラインとの再比較、他の3データセット、アブレーション、有意差検定は対象外である。また、City of Chicagoのraw tripデータからの再構築ではなく、著者配布の処理済み`chicago20_23.npz`を固定して使用する。

## 前実験からの変更

初回の一体実行（warm-up→test）はオンライン学習率が約1.16×10^-13に減衰し、MAE 0.890382・RMSE 3.997956・WMAPE 45.7402%となった。これは既存rawのfrozen結果と一致する。今回の再実行ではbest epoch 25のwarm-up checkpointを保持し、rawで良好だった`train_only→mode=online`経路に合わせてonlineのみ再実行する。旧ログ・指標は`attempts/`へ退避する。論文の5 seed平均・標準偏差による最終評価は後続実験を含めて行う。

## 仮説

1. 論文と同じデータ分割と主要ハイパーパラメータを用いれば、3指標は論文報告値に近い値になる。
2. LSLだけを周期的に更新することで、約3年間のonline期間に含まれる長期的な需要変化へ追従できる。
3. `rmse`（全サンプル・ホライズン・地点をまとめたglobal RMSE）は論文値と比較可能であり、`sample_rmse`は集約方法の違いを確認する補助指標になる。

## 手法

### データと分割

- データ：Chicago-T（2020-01-01〜2023-12-31、15分間隔、77地点）。
- 入力：過去12ステップ（3時間）の需要。
- 教師：未来12ステップ（3時間）の需要。
- 時系列分割：train 20%、validation 5%、online 75%。シャッフルして分割しない。
- 標準化：train期間だけから全地点共通の平均と標準偏差を推定し、全期間へ適用する。
- 境界直後の入力には直前区間の履歴を使うが、教師12ステップは必ず対象区間内に収める。
- 隣接行列：配布済み`adj_chicago.npy`を使用し、自己ループを加えた順方向・逆方向の遷移行列を固定支持行列とする。

### モデルとwarm-up

- バックボーン：Graph WaveNet（残差・dilationチャネル32、8層、適応的隣接行列あり）。
- LSL：77地点ごとに独立した`32 → 4 → 32`のボトルネック層を、時空間バックボーンの前段へ置く。
- warm-upではバックボーンとLSLを同時に学習する。
- 学習・online更新の損失：逆標準化後の元スケールMAE。early stopping用のvalidationはrawと同じ標準化スケールMAE。
- optimizer：AdamW、初期学習率0.001、weight decay 0.01。
- batch size 32、最大150 epoch、early stopping patience 10、seed 42、決定論的実行を用いる。
- validation MAEが最良のcheckpointをonline評価へ引き継ぐ。
- online開始時に新しいmodel/optimizer/SMBを構築し、このcheckpointのモデル重みとscalerだけを復元する。validation 7,002件を新しいSMBへ1回投入する。

### online適応

- onlineは時系列順、batch size 1で処理する。
- 予測長と同じ12ステップ後に教師を解禁し、現在時刻で未観測の未来値を更新へ使わない。
- SMB容量は1,000件（GPU常駐）、1回の更新に使うEpisodic Memoryは8件、更新は各ステップ1回とする。
- 1週間（672ステップ）Awake、1週間Hibernateを交互に繰り返す。
- Awake中はGraph WaveNet本体を凍結してLSLだけを更新し、Hibernate中はパラメータを更新しない。
- AwakeからHibernateへ移る境界でSMBをリセットし、休眠中に到着したサンプルを次のAwakeで利用する。
- online optimizerはrawの`mode=online`同様、学習率0.001・空のstateで新規作成する。warm-upの減衰済みoptimizerは引き継がない。

### 評価と比較

- 主指標：MAE、global RMSE、WMAPE。rawと同様に全予測・正解をCPUに連結してNumPyで一括集計する。`metrics.json`のWMAPEは比率なので、論文との比較時は100倍して百分率にする。
- 補助指標：sample-wise RMSE、12ホライズン別MAE/RMSE/WMAPE、online更新回数、Awake/Hibernate件数、処理時間。
- 論文比較には`rmse`を使う。`sample_rmse`は各サンプル内の12ホライズン×77地点でRMSEを求め、その後サンプル間で平均した値であり、論文表2との直接比較には使わない。
- `exp00`の単一seed結果について論文値からの絶対差と相対差を記録する。後続の計5 seedでは平均±標準偏差を計算し、論文の集計単位に合わせる。
- 処理時間はGPUや実行環境に依存するため参考値とし、論文との直接的な速度再現判定には用いない。

### 実行手順

1. 実行前のcommit、dirty状態、設定値を記録する。
2. `uv run --offline --no-sync python src/run_dol.py exp00 --reuse-checkpoint`で既存checkpointからonlineだけ実行する。
3. checkpointのSHA-256が不変で、`config.json`、`git_commit.txt`、`run.log`、`metrics.json`が新試行を記録し、旧記録が`attempts/`に退避されたことを確認する。
4. `report.md`に論文値との差、学習曲線、更新回数、失敗や実装差を記録する。
5. 同じ設定でseedだけを変えた後続実験を行い、5 seed集計で最終的な再現可否を判断する。

## 成功条件

- 既存warm-up checkpointを変更せず、全online区間が例外や非有限値なしで完了し、`status`が`completed`になる。
- onlineの105,181予測窓を時系列順に処理し、予測・更新・AH件数が`metrics.json`へ記録される。
- MAE、global RMSE、WMAPEの各単一seed結果が、論文報告値から相対5%以内に入ることを暫定的な数値再現の成功とする。
- 5 seed完了後は平均±標準偏差を論文値と比較する。5%を超える場合も直ちに失敗とは断定せず、データ、乱数、optimizer状態、評価集約、公開実装との差を切り分けて報告する。
