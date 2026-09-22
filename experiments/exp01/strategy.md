# exp01：Chicago-T における DOL の5 seed再現実験（計画・未実行）

## 実験目的

[DOL論文 Table 2](https://www.ijcai.org/proceedings/2025/0372.pdf) の Chicago-T における DOL の MAE 0.72±0.00、RMSE 2.06±0.02、WMAPE 36.80%±0.19% と、同じデータ・モデル・学習・オンライン評価条件で得た5試行の平均と標準偏差を比較する。exp00の単一seed（42）では MAE 0.698637、global RMSE 2.215612、WMAPE 35.8900%であり、特にRMSE差が複数seedでも残るか確認する。

論文は「異なるseedで5回」と記すがseed番号自体は公開していない。そのため **42, 43, 44, 45, 46 は本実験で事前指定するseed** であり、著者の乱数列まで厳密に再現するものではない。GTX 3090以外のGPUでの実行時間も論文との直接比較対象にしない。

## 前実験からの変更と固定条件

- exp00はseed 42の1試行。exp01では42〜46の5試行を同一手順で**それぞれwarm-upから独立に実行**する。exp00のcheckpointと結果は流用せず、exp00は参照値として保持する。
- 変更する実験条件は乱数seedだけとする。著者配布の同じChicago-T NPZ・隣接行列を使い、両ファイルのSHA-256を記録する。時系列のtrain/validation/onlineは20%/5%/75%、入力・予測は各12ステップ、77地点。
- Graph WaveNetとLSL（チャネル32、ボトルネック4）、AdamW学習率0.001、最大150 epoch、patience 10、SMB容量1,000、EM 8件、Chicago-Tの1週間672ステップごとのAwake/Hibernate、12ステップの教師遅延をexp00から変えない。より細かな設定は各seedの`config.json`に保存する。
- 各試行のwarm-upは最良validation checkpointを選ぶ。onlineはそのモデル重みとscalerを引き継ぎ、optimizerとSMBを新規初期化し、validationをSMBへ1回投入する。これはexp00でrawと結果が一致した経路に合わせる。
- 全予測・正解をオンライン評価の最後に集約し、主指標をMAE、**global RMSE**、WMAPEとする。**sample-wise RMSEも各seedで計算し、5 seedの平均±標準偏差まで報告する**。sample-wise RMSEとホライズン別指標は補助指標で、論文のRMSEには代用しない。

## 実装・記録方法（実装済み・実験未実行）

既存の`src/run_dol.py exp01`はseed 42を**1回だけ**実行するため使用しない。5 seed用の`src/run_exp01.py`と`src/dol/multiseed.py`を追加し、GPUを使わない`tmp/test_multiseed.py`で成果物分離・集計・失敗再開を検証した。この一時テストはGit対象外。GPU実験はまだ開始していない。

1. 実行器は`ExperimentConfig.warmup.seed`へ42〜46を順に渡す。各試行でモデル・データローダー・SMBの乱数を設定し、warm-upとonlineを独立に実行する。exp00用の既定値・CLIの挙動は変えない。
2. 各試行の成果物は`experiments/exp01/seeds/seed42/`〜`seed46/`へ分離する。各`metrics.json`にはsample-wise RMSEも含む。完了済みseedは検証してスキップし、失敗・中断したseedは`--retry-failed`を明示した場合だけ旧成果物を`attempts/`へ退避してやり直す。checkpoint、予測配列、退避済み試行はGit対象外とする。
3. 直下の`config.json`へ共通設定・seed一覧・データSHA-256を、`metrics.json`へ各seedの値、5試行完了後の平均・標本標準偏差、論文値との相対差と5%判定を記録する。`run.log`は試行開始・終了を記録し、詳細ログはseed別に保存する。`report.md`には論文・exp00との比較と考察を人が記入する。
4. コード・計画をcommitして作業ツリーをきれいにしてから、下記コマンドでGPU実験を開始する。開始時のrevisionを5試行に共通して記録し、`git_commit.txt`の`dirty: false`を確認する。

## セットアップと実行

リポジトリのルートで`uv sync`を実行し、[著者のデータ配布先](https://drive.google.com/drive/folders/1fpHzT_jyoHhr2uQA10-v3JTcoHCVc8IU?usp=drive_link)から`chicago20_23.npz`と`adj_chicago.npy`を取得して`datasets/chicago-t/`へ配置する。形式は[datasets/chicago-t/README.md](../../datasets/chicago-t/README.md)を参照。初回実行前にコード・計画をcommitし、`git status --short`が空であることを確認する。`run_dol.py exp01`はseed 42の単発実行になり、5 seedの集計を行わないため使わない。

```bash
uv run --offline --no-sync python src/run_exp01.py
# 失敗・中断した試行の旧成果物を保存して再試行する場合のみ
uv run --offline --no-sync python src/run_exp01.py --retry-failed
```

端末を閉じても継続させる場合は、リポジトリのルートから次を実行する。標準出力・エラーはリポジトリ外へ送り、進捗は`experiments/exp01/run.log`、詳細は各seedの`run.log`で確認する。

```bash
nohup uv run --offline --no-sync python src/run_exp01.py \
  > /tmp/dol-exp01-nohup.log 2>&1 < /dev/null &
echo $!
```

## exp01の保存構成（実装後の予定）

```text
experiments/exp01/
├── strategy.md              実行前の目的・条件・判定基準
├── config.json              共通設定とseed一覧
├── git_commit.txt           5試行に共通するコード版
├── run.log                  5試行の開始・終了・失敗の記録
├── metrics.json             5 seedの個別値と平均±標準偏差
├── report.md                論文・exp00との比較と考察
└── seeds/
    ├── seed42/
    │   ├── config.json      当該seedを含む実行設定
    │   ├── git_commit.txt   試行時のコード版とdirty状態
    │   ├── run.log          warm-up・onlineの進捗
    │   ├── metrics.json     MAE・global/sample-wise RMSE・WMAPE等
    │   ├── checkpoints/
    │   │   └── warmup.pt    最良warm-up重み（Git対象外）
    │   └── attempts/        失敗・中断試行の退避先（Git対象外、任意）
    ├── seed43/              seed42と同じ構成
    ├── seed44/              seed42と同じ構成
    ├── seed45/              seed42と同じ構成
    └── seed46/              seed42と同じ構成
```

現在は直下のファイルが未実行の雛形で、`seeds/`以下は未作成。実行時に直下の`config.json`・`metrics.json`・`run.log`を更新する。各seedのcheckpointは保存するがGitに載せず、集計結果とログを公開対象にする。

## 集計・比較方法

- 各seedのonline期間全体についてMAE、global RMSE、sample-wise RMSE、WMAPEを1値ずつ得る。sample-wise RMSEは各予測窓の12ホライズン×77地点でRMSEを求め、その窓間平均とする。WMAPEは`metrics.json`の比率を100倍して%表示する。4指標それぞれについて5試行の算術平均と標本標準偏差（`ddof=1`）を計算する。論文側の標準偏差の計算規約と丸め前の値は不明なので、標準偏差の完全一致は判定条件にしない。
- 主比較はMAE・global RMSE・WMAPEの**seed間平均±標準偏差**対論文Table 2とする。sample-wise RMSEも同じ集計で併記するが、論文には対応する値がないため数値再現の判定には使わない。5試行を連結して1つのglobal RMSEを求めたり、sample-wise RMSEで置き換えたりしない。個々のseed結果とexp00 seed 42との差も併記する。
- 事前判定として、3つの平均値がいずれも論文平均から相対5%以内なら「数値的に近い」とする。外れた場合は指標別の差をそのまま報告し、原因未確定のまま再現成功とは書かない。論文の表示値は丸め済みのため、この閾値は本実験の暫定基準である。

## 仮説と成功条件

- seed間でもMAEとWMAPEは論文値に近く、exp00で見られたglobal RMSEの上振れは残る可能性がある。結果に応じて仮説を棄却・修正する。
- 5 seedすべてが同じコード・データ・固定設定（seed以外）で完了し、オンライン窓数と更新件数が記録されることを実行面の成功条件とする。
- 失敗やOOMが起きた場合は欠測を平均から黙って除外せず、理由と再試行を報告する。5件未満なら論文の「5回反復」として比較しない。
