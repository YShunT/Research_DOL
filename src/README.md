# DOLを読みやすく再実装する

このディレクトリは、DOL（Distribution-Aware Online Learning）を、論文の構成とコードの責務が対応するように再実装した学習用コードである。

- `raw/DOL_original`を参照専用の互換基準とし、この再実装からはimportしない。
- 公開実装の既定値、初期化順、学習、SMB、online更新の挙動を保つ。
- データ、モデル、warm-up、online、評価を責務ごとのmoduleへ分け、読みやすくする。
- CLIの`argparse`は使わず、dataclassの設定をPythonから明示的に渡す。
- module内部で`.cuda()`を呼ばない。モデル、入力、グラフ支持行列を実行側で同じdeviceへ置く。
- importしただけでは学習を開始しない。[`run_dol.py`](run_dol.py)または[`run_exp01.py`](run_exp01.py)を直接実行した場合だけ実験を開始する。

数式、Tensor形状、各クラスの役割、公開実装との差も本READMEにまとめる。

## 1 実装された範囲

DOLの処理全体を次の順で実装している。

```text
Chicago-T NPZ
  ↓ 需要列の抽出
訓練20% / 検証5% / オンライン75%
  ↓ 訓練20%だけで平均・標準偏差を計算
過去12点 → 未来12点の窓
  ↓
入力埋め込み Conv2d(1 → 32)
  ↓
77地点に独立したLSL: 32 → 4 → 32
  ↓ 残差加算
Graph WaveNet: gated TCN + 拡散GCN × 8層
  ↓
12ホライズン予測
  ↓
warm-up: 全329,252パラメータを学習
  ├─ 各epochの検証標本をwarm-up用SMBへ投入
  └─ 最良checkpointを保存
  ↓ warm-up用SMB・optimizerは引き継がない
online用model/optimizer/SMBを新規作成し、checkpointのmodel/scalerを復元
  ↓ validation 7,002窓を新しいSMBへ1回投入
online: LSLの22,484パラメータだけ更新
  ├─ 12ステップ遅延した教師
  ├─ 容量1,000のStreaming Memory Buffer
  ├─ 8標本のEpisodic Memory
  └─ 1週間Awake / 1週間Hibernate
  ↓
MAE / global RMSE / WMAPE（全体・ホライズン別）＋sample-wise RMSE（補助）
```

## 2 ディレクトリ構成

```text
src/
├── README.md
├── run_dol.py
├── run_exp01.py
└── dol/
    ├── artifacts.py
    ├── multiseed.py
    ├── config.py
    ├── pipeline.py
    ├── data/
    │   ├── scaler.py
    │   ├── windows.py
    │   └── chicago.py
    ├── graph/
    │   └── supports.py
    ├── layers/
    │   ├── LSL.py
    │   ├── graph_convolution.py
    │   └── graph.py
    ├── models/
    │   └── dol.py
    ├── training/
    │   ├── losses.py
    │   └── warmup.py
    ├── online/
    │   ├── delay.py
    │   ├── replay.py
    │   ├── schedule.py
    │   └── adapter.py
    └── evaluation/
        └── metrics.py
```

## 3 各Pythonファイルの役割

### 3.1 実行と設定

#### [`run_dol.py`](run_dol.py)

DOL実験の入口である。全体の流れは次のようになる。

```python
from run_dol import main

main(experiment_name="exp02")  # warm-up後、online用にmodel/optimizer/SMBを新規構築
```

引数なしなら次の`exp<ii>`を割り当て、`exp02`のような引数があれば未実行の同名ディレクトリを使う。`--reuse-checkpoint`付きなら完了済み番号のwarm-up checkpointを残し、旧ログ・指標を`attempts/`へ退避してonlineだけ再実行する。`if __name__ == "__main__":`の内側でだけ`main()`を呼ぶため、他のコードからimportしても実験は始まらない。

#### [`run_exp01.py`](run_exp01.py)

Chicago-Tのseed 42〜46を独立に実行する入口。`--retry-failed`を付けた場合だけ失敗・中断した試行を退避して再実行する。単発用の`run_dol.py exp01`では5 seedにならない。

#### [`dol/multiseed.py`](dol/multiseed.py)

exp01の順次実行と集計を担当する。実行開始時のGit revisionと入力データのSHA-256を固定し、各seedの設定、checkpoint、処理件数、指標を検証する。MAE・global RMSE・sample-wise RMSE・WMAPEをseed間で平均・標本標準偏差に集計し、論文値との相対差も記録する。完了済みseedはスキップし、失敗・中断試行の旧成果物は`attempts/`へ退避する。

#### [`dol/artifacts.py`](dol/artifacts.py)

実験名と成果物を管理する。単発の`exp<ii>`に加えてexp01内の`seed<ii>`にも対応する。ローカルに`experiments/_template`があれば新しい単発実験へ複製するが、Git対象外であり、なくても必要な文書を生成する。seed別試行では直下に計画・レポートを置くため、文書の自動生成を抑える。`config.json`、`git_commit.txt`、`run.log`、`metrics.json`を安全に作成する。既に実行結果があるディレクトリは通常上書きしない。checkpoint再利用時は旧記録だけ退避する。予測配列を保存する設定では、Git対象外の`arrays/predictions.npy`と`arrays/targets.npy`も作る。

#### [`dol/config.py`](dol/config.py)

コマンドライン引数の代わりとなる不変dataclassを定義する。

|クラス|責務|主な既定値|
|---|---|---|
|`DataConfig`|ファイル、ノード数、窓長、分割|77地点、12→12、20/5/75%|
|`ModelConfig`|Graph WaveNet、LSL、グラフ演算|32チャネル、LSL幅4、8層|
|`WarmupConfig`|初期学習|batch 32、150 epoch、patience 10|
|`OnlineConfig`|SUM|SMB 1,000、EM 8、Awake/Hibernate各1週|
|`RuntimeConfig`|deviceと再現性|`cuda:0`、deterministic|
|`ArtifactConfig`|実験保存|`experiments/`、予測配列は保存しない|
|`ExperimentConfig`|上記を1実験へ統合|全既定値|

`ModelConfig.receptive_field`は構造から受容野を計算する。既定値は13であり、過去12点入力の左へ1点paddingする。

#### [`dol/pipeline.py`](dol/pipeline.py)

各モジュールを接続するcomposition rootである。

|関数|行うこと|
|---|---|
|`build_components`|データ、隣接行列、支持行列、モデルを構築|
|`build_dataloaders`|trainだけshuffleし、validation/onlineを時系列順にする|
|`run_warmup`|`WarmupTrainer.fit`を呼ぶ|
|`load_warmup_checkpoint`|モデルとscalerを復元|
|`run_online_evaluation`|CLIからはvalidationを1回投入する指定で呼び、1窓ずつ適応・予測して指標を集計|

モデルの式や学習則はここへ書かず、対応するクラスへ委譲している。このファイルを読むと、DOL全体で何がどの順に呼ばれるかが分かる。

### 3.2 データ

#### [`dol/data/scaler.py`](dol/data/scaler.py)

`GlobalStandardScaler`を定義する。warm-up訓練期間の全地点・全時刻から単一の平均$\mu$と標準偏差$\sigma$を求める。

```text
x_normalized = (x - mean) / std
x_original   = x_normalized * std + mean
```

NumPyとPyTorch Tensorの両方を扱う。Tensorの逆変換は勾配を切らないため、元スケールのMAEからモデルへ逆伝播できる。

#### [`dol/data/windows.py`](dol/data/windows.py)

`compute_split_boundaries`が時刻軸をtrain/validation/onlineへ分割する。`TimeWindowDataset`は連続系列から、過去$L$点の入力と未来$H$点の教師を作る。

validationとonlineの最初の予測では、入力履歴だけ直前の分割を参照する。教師窓は必ず現在の分割内に収まるため、予測対象の重複や未来漏洩はない。

#### [`dol/data/chicago.py`](dol/data/chicago.py)

Chicago-T固有の読込みを担当する。

1. `chicago20_23.npz`の`data`を読む。
2. 最終列の乗車需要だけを公開実装と同じ`float64`で取り出す。
3. 負値、非有限値、ノード数を検査する。
4. train部分だけでscalerをfitする。
5. 3つの`TimeWindowDataset`を`ChicagoDataBundle`へまとめる。

NPZはTimestampを含むobject配列なので、配布済みの信頼できるファイルに限り`allow_pickle=True`で読む。

### 3.3 グラフ

#### [`dol/graph/supports.py`](dol/graph/supports.py)

固定隣接行列$A$からGraph WaveNet用の支持行列を作る。

```text
A_with_self_loop = A + I
P_forward  = row_normalize(A_with_self_loop)
P_backward = row_normalize(A_with_self_loop.T)
```

`load_adjacency`は`.npy`を`float32` Tensorとして読み、正方行列・ノード数・有限値を検査する。

#### [`dol/layers/graph_convolution.py`](dol/layers/graph_convolution.py)

実行パイプラインが使用するグラフ層である。

- `NodePropagation`：公開実装と同じ$XA$を`einsum`で計算する。
- `DiffusionGraphConvolution`：各支持行列の1次・2次拡散を連結し、1×1 Convで統合する。
- `AdaptiveAdjacency`：$\operatorname{softmax}(\operatorname{ReLU}(E_1E_2))$を学習する。

`output[...,w] = sum_v input[...,v] * support[v,w]`として、特徴のノード軸へ支持行列を右から掛ける。

#### [`dol/layers/graph.py`](dol/layers/graph.py)

理論を自分で追って書いた学習用の簡潔版である。以前のスペル`DifuusionGraphConvolution`へのaliasも持つ。完成したDOLパイプラインは、検査とdocstringが詳しい`graph_convolution.py`を使用する。このファイルは既存コードとして残している。

### 3.4 地点固有学習器

#### [`dol/layers/LSL.py`](dol/layers/LSL.py)

- `NodeAdapter`：1地点の特徴を`32 → 4 → 32`で補正する。
- `LocationSpecificLearner`：地点数だけ独立した`NodeAdapter`を`ModuleList`へ持つ。

LSLは補正量だけを返す。残差加算

```python
x = embedded + location_specific(embedded)
```

はモデル側で行う。Chicago-Tでは1地点292、77地点で22,484パラメータである。

### 3.5 DOL予測モデル

#### [`dol/models/dol.py`](dol/models/dol.py)

`GraphWaveNetBlock`は1層分の処理をまとめる。

```text
residual
  ├─ temporal filter Conv → tanh ─┐
  └─ temporal gate Conv   → sigmoid ─ elementwise product
                                      ├─ skip 1×1 Conv
                                      └─ diffusion GCN → residual add → BatchNorm
```

`DOLForecastModel`は次を接続する。

1. `input_projection`：1特徴を32チャネルへ埋め込む。
2. `location_specific`：地点別補正を作り、入力埋め込みへ足す。
3. `adaptive_adjacency`：3番目の支持行列を学習する。
4. `blocks`：dilation 1,2を4回繰り返す8層。
5. `output_projection`：skip 256→512→12ホライズン。

`freeze_for_online_adaptation()`は全パラメータを凍結した後、LSLだけ`requires_grad=True`にする。さらに`eval()`へ置くため、オンライン更新中もBatchNormの統計とDropoutが固定される。

Chicago-T既定設定で確認した値は次のとおりである。

|項目|値|
|---|---:|
|入力shape|`(B, 12, 77)`|
|出力shape|`(B, 12, 77)`|
|受容野|13|
|総パラメータ|329,252|
|オンライン更新対象|22,484|
|更新対象比|6.83%|

公開実装と同じく、GCN経路でforwardに使われない1×1 `residual_convs`も登録する。これによりパラメータ数だけでなく、同一seedでの後続層の初期化順も一致する。

### 3.6 warm-up

#### [`dol/training/losses.py`](dol/training/losses.py)

`original_scale_mae`は標準化を戻した需要値でMAEを計算する。Chicago-Tの0は有効な「需要なし」であるためmaskしない。

#### [`dol/training/warmup.py`](dol/training/warmup.py)

`WarmupTrainer`は全パラメータをAdamWで学習する。

- train：勾配計算とoptimizer更新。
- validation：標準化スケールのMAEを評価し、同時に標本をSMBへ投入する。
- early stopping：検証MAEが改善しなければpatience後に終了。
- checkpoint：モデル、optimizer、scaler、設定、seed、epochを保存。
- fit終了：検証MAEが最良だった重みを必ず復元。

公開実装と同じく、warm-upの各epochでvalidationをSMBへ追加する。ただしCLIはrawの成功した`train_only→mode=online`経路に合わせ、online用の新しい部品を作るためこのSMBを引き継がない。online開始前に空のSMBへvalidationを1回投入する。

### 3.7 online/SUM

#### [`dol/online/delay.py`](dol/online/delay.py)

`DelayedSupervisionQueue`が、時刻$s$の未来$H$点教師を$s+H$まで隠す。オフラインDatasetから教師が同時に渡されても、online optimizerは解禁前に参照できない。

#### [`dol/online/replay.py`](dol/online/replay.py)

`ReservoirReplayBuffer`がStreaming Memory Buffer（SMB）を実装する。

- 最大1,000件をrawと同じくGPUに常駐（CPU設定時はCPU）。
- $k$件目の到着標本を容量$M$へ確率$M/k$で採用。
- `sample(8)`がその時点のEpisodic Memory（EM）となる。
- 休眠開始時に`clear()`し、古い周期の標本を捨てる。

#### [`dol/online/schedule.py`](dol/online/schedule.py)

`AwakeHibernateSchedule`が公開実装のcountとmoduloによる周期判定を明示的に管理する。既定値は1週間672ステップであり、最初のAwakeがstep 0を含む673ステップになる公開実装の境界も維持する。

#### [`dol/online/adapter.py`](dol/online/adapter.py)

`OnlineAdapter`がSUM全体を統合する。1ステップで必ず次の順を守る。

```text
online開始前: 新しいSMBへvalidationを1回だけ投入
  ↓
成熟した教師を解禁
  → SMBへ追加
  → AwakeならEMを抽出してLSL更新
  → 現在窓を予測
  → 現在教師を遅延キューへ格納
  → AH状態を進める
```

`OnlineStepResult`には予測、真値、位相、更新MAE、解禁標本数、SMBサイズが入る。研究用のログを追加するときはこの値を保存すればよい。

### 3.8 評価

#### [`dol/evaluation/metrics.py`](dol/evaluation/metrics.py)

`compute_forecast_metrics`がrawと同じくonline全予測・正解をCPUに連結し、NumPyで一括集計する。`ArtifactConfig.save_predictions=True`のときだけ、集計後の配列をnpyにも保存する。Chicago-Tの約10.5万窓では予測・正解の連結済み配列だけで約0.78 GBを使い、連結中と指標計算中の一時配列を含むRAMピークは数GBになる。

- 全体MAE、global RMSE、sample-wise RMSE、WMAPE。
- 1–12ステップ先それぞれのMAE、RMSE、WMAPE。
- 評価した総要素数。

需要が0の地点を含むため、単純なMAPEではなく分母を全真値の絶対値和とするWMAPEを使う。

## 4 実行方法

### 4.1 環境

リポジトリルートを1つのuvプロジェクトとして管理する。プロジェクトルートには次がある。

|ファイル・ディレクトリ|役割|
|---|---|
|`pyproject.toml`|Pythonの範囲、依存関係、build設定を宣言|
|`uv.lock`|推移依存を含む解決結果を固定|
|`.python-version`|rawの検証環境に合わせてPython 3.11を指定|
|`.venv/`|uvが同期するローカル仮想環境|

実行時依存は、公開実装に合わせたPyTorch 2.1.1、NumPy 1.xと、Chicago-TのNPZ内のTimestamp復元に必要なpandas 2.xだけを宣言している。

最初の構築または依存更新後は、研究ディレクトリで同期する。

```bash
uv sync
```

依存の追加・削除は`pip install`ではなく、プロジェクトルートで次のように行う。

```bash
uv add パッケージ名
uv remove パッケージ名
```

`uv add`と`uv remove`は`pyproject.toml`と`uv.lock`を同時に更新する。通常の実行には`uv run`を使う。シェルで仮想環境をactivateする必要はない。

VS Codeでは、Python interpreterとして次を選ぶ。

```text
.venv/bin/python
```

本番GPUでは既定の`cuda:0`を使用する。CPUで構造だけ確認するときは設定を明示する。

```python
from dol.config import ExperimentConfig, RuntimeConfig

config = ExperimentConfig(runtime=RuntimeConfig(device="cpu"))
```

実装はCPU専用に分岐していない。同じmoduleを任意のCUDA deviceへ配置できる。

### 4.2 全warm-upとonline評価

次のコマンドは、既存番号の次に当たる実験ディレクトリを作って実行する。

```bash
uv run python src/run_dol.py
```

未実行の番号（例：`exp02`）へ単発の結果を保存する場合は次のように指定する。実行済みの`exp00`は通常の実行では上書きできない。`exp01`は5 seed再現計画用に予約している。

```bash
uv run python src/run_dol.py exp02
```

実行済みディレクトリは通常上書きしない。既存checkpointから同じ番号のonlineだけをやり直す場合は、次を実行する。旧ログ・指標・設定は`attempts/`へ退避され、checkpointは保持される。

```bash
uv run --offline --no-sync python src/run_dol.py exp00 --reuse-checkpoint
```

`--no-sync`は既に同期済みの`.venv`を使う指定で、uvキャッシュ削除後に`--offline`で実行する場合に必要になることがある。

exp01の5 seed（42〜46）は単発CLIではなく`src/run_exp01.py`で順番に実行する。実験前にコード・計画をcommitし、作業ツリーをきれいにする。途中まで完了した場合は完了済みseedを検証してスキップし、失敗・中断したseedだけを再試行するには`--retry-failed`を付ける。各seedの`metrics.json`にsample-wise RMSEも保存し、exp01直下の`metrics.json`に4指標の平均・標本標準偏差を保存する。

```bash
uv run --offline --no-sync python src/run_exp01.py
uv run --offline --no-sync python src/run_exp01.py --retry-failed
```

### 4.3 `exp<ii>`のファイルとコードの関係

```text
run_dol.py
  ├─ artifacts.create_experiment_run()
  │    ├─ ローカル_templateがあればexp<ii>へ複製
  │    ├─ config.json
  │    ├─ git_commit.txt
  │    └─ run.log
  ├─ pipeline.run_warmup()（checkpoint再利用時は省略）
  │    └─ checkpoints/warmup.pt
  ├─ 新しいmodel/optimizer/SMBを構築しcheckpointを読み込む
  ├─ validationをSMBへ1回投入
  ├─ pipeline.run_online_evaluation()
  │    ├─ evaluation/metrics.py → metrics.json
  │    └─ save_predictions=True → arrays/*.npy
  └─ 人が結果を確認
       ├─ figures/
       └─ report.md
```

|成果物|対応するコード|用途|
|---|---|---|
|`strategy.md`|ローカル雛形から複製、なければ自動生成|実行前の仮説と変更点を人が記述|
|`config.json`|`ExperimentConfig`を`artifacts.py`が直列化|使用条件の再確認|
|`git_commit.txt`|`artifacts.py`|コードrevisionと未commit変更の有無|
|`run.log`|`run_dol.py`のlogger|warm-up各epoch、online 1,000件ごとの進捗、例外|
|`metrics.json`|`WarmupResult`と`OnlineEvaluation`|warm-up履歴、MAE/RMSE/WMAPE、AH件数、所要時間|
|`checkpoints/warmup.pt`|`WarmupTrainer.save_checkpoint`|最良epochのモデル・optimizer・scaler・設定|
|`arrays/predictions.npy`|`OnlineEvaluation.predictions`|任意保存の`(online窓数, H, N)`予測|
|`arrays/targets.npy`|`OnlineEvaluation.targets`|予測と対応する同shapeの正解|
|`figures/`|後段の分析|レポート用の図|
|`report.md`|人|結果の解釈と次の実験|

`git_commit.txt`の`commit`は実行開始時のGitコミットID、`dirty`はその時点に未コミット変更があったかを表す。これはモデル重みやGitの署名ではなく、結果を生成したコード版をたどるための記録である。exp01は開始時の同一revisionを各seedへ引き継ぐため、実験中にログ・指標が増えてもseed間で記録が変わらない。

`run.log`は進捗と終了値を検証できるようGitで追跡する。`strategy.md`、`config.json`、`git_commit.txt`、`metrics.json`、必要な図、`report.md`も実験記録としてcommitする。checkpointとnpyは容量が大きいため除外する。

全予測を保存したい実験だけ、設定を明示してPythonから実行する。

```python
from dol.config import ArtifactConfig, ExperimentConfig
from run_dol import main

config = ExperimentConfig(
    artifacts=ArtifactConfig(save_predictions=True),
)
main(experiment_name="exp02", config=config)
```

Chicago-T既定条件では予測と正解を合わせて大きな容量になるため、`save_predictions=False`が既定値である。

### 4.4 Pythonから段階別に実行

```python
from pathlib import Path

from dol.config import ExperimentConfig
from dol.pipeline import (
    build_components,
    load_warmup_checkpoint,
    run_online_evaluation,
    run_warmup,
)

config = ExperimentConfig()
components = build_components(config)

# warm-upでbest checkpointを保存する
run_warmup(components, Path("experiments/exp00/checkpoints/warmup.pt"))

# rawのmode=onlineと同じく、新しいoptimizer/SMBと乱数状態から始める
del components
components = build_components(config)
load_warmup_checkpoint(
    components,
    Path("experiments/exp00/checkpoints/warmup.pt"),
)
online = run_online_evaluation(components, seed_validation_before_online=True)
print(online.metrics)
```

動作確認だけなら、onlineの先頭件数を制限できる。

```python
online = run_online_evaluation(components, max_steps=100)
```

## 5 公開実装の挙動を保った整理

|公開実装の状態|この再実装|
|---|---|
|巨大な`args`へ途中で属性を追加|意味別の不変dataclass|
|model内部でdeviceを保持・TensorをCUDAへ作成|実行側で`.to(device)`|
|validation関数が各epochでbufferも更新|副作用を明示した引数に分け、同じタイミングでSMBを更新|
|rawの`mode=train`では減衰後の学習率をonlineへ引き継ぐ|rawの`train_only→mode=online`に合わせ、新規optimizerを学習率0.001で作る|
|AdamWへ`weight_decay`を渡さず既定0.01が使われる|設定へ0.01を明記して渡す|
|offline Datasetが返す未来教師をその場で保持|`DelayedSupervisionQueue`でHステップ隠す|
|broadな`except`でskip初期化|`None`で明示的に初期化|
|モデル、学習、オンライン状態が`Exp`へ集中|責務ごとのmoduleへ分割|
|checkpointがmodel重みだけ|optimizer、scaler、設定、seedも保存|
|全予測を配列へ保存して指標計算|同じくCPUに全件保持してNumPyで一括集計し、設定時だけnpyを保存|

入力埋め込み、地点別LSL、Graph WaveNet、$XA$型グラフ伝播、未使用`residual_convs`を含む初期化順、warm-up、SMB/EM、AH境界は公開実装に合わせる。onlineのoptimizerはcheckpoint-only経路と同じく新規作成する。

## 6 実装上の重要な判断

### 6.1 0需要をmaskしない

Chicago-Tの0は欠損ではなく実際の需要0である。この再実装は全要素を含むMAEを使う。公開実装のクラス`MaskedMAE`も既定の`null_val=np.nan`ではNaNだけを除き、0を含めるため、この点の損失定義は一致する。関数名だけを見て「0を除外する」と解釈してはいけない。

### 6.2 online更新ではmodelをeval modeにする

更新対象はLSLだけだが、損失の勾配は固定backboneを通ってLSLへ戻る。`eval()`は勾配を無効にしない。BatchNormのrunning statisticsとDropoutだけが固定される。公開実装も`test()`冒頭で`model.eval()`を呼んだままオンライン逆伝播するため、この挙動と一致する。

### 6.3 教師はHステップ後に利用する

未来12点のうち最初の1点だけが観測された段階では、12ホライズン損失を計算できない。全12点が観測されるまで待ち、1つの完全な教師窓としてSMBへ入れる。

### 6.4 予測は休眠中も止めない

Hibernateは学習更新を止める状態であり、推論停止ではない。全online窓で予測・評価を続ける。

## 7 論文・公開実装との照合結果

2026年9月19日に、DOL論文のMethodology・Algorithm 1・Experimental Settings、`raw/DOL_original`、この実装を項目別に照合した。

### 7.1 方法の中心部分

次は論文と一致する。

- warm-up 25%（train 20%＋validation 5%）とonline 75%の時系列分割。
- $L=H=12$、LSLの$32\to4\to32$、入力埋め込み後・ST module前への配置。
- warm-upでは全体、AwakeではLSLだけを更新する。
- SMB容量1,000、EMサイズ8、reservoir sampling。
- 1週Awake・1週Hibernate、Hibernate開始時のSMB reset。
- $H$ステップ後に完全な教師窓を利用する因果順序。
- validation標本によるonline開始時のSMB初期化。
- AdamW、学習率0.001、最大150 epoch、patience 10。
- MAE、RMSE、WMAPEによる評価。

### 7.2 公開コード互換の実行詳細

- trainの端数batchを捨て、各epoch後に学習率を0.5倍する。
- Python、NumPy、PyTorch、CUDAのseed offsetと、cuDNN deterministic・TF32無効の既定設定を保つ。
- CLI実験はrawの`train_only→mode=online`経路を使い、新しいSMBへvalidationを1回投入する。
- online optimizerを学習率0.001・空のstateで新規作成する。rawの`mode=train`一体実行とは異なる。現行の`build_components`は`WarmupConfig.learning_rate`と`weight_decay`で新しいoptimizerを作る。
- AH位相は公開実装のcount更新順とmodulo条件を保つ。
- future targetは、公開実装のrecall Tensorと同じ$H$ステップ後に解禁する。
- forwardで使われない`residual_convs`も、パラメータ数と初期化順の互換のため登録する。

### 7.3 完全再現ではない部分

- 論文の定義には外部要因$E$があるが、公開コードもこの再実装も日時特徴を予測器へ入力しない。AH周期は経過ステップ数で管理する。
- 固定支持行列は公開実装のSciPyではなくPyTorchで同じ式とdtype変換順を計算する。配布Chicago-Tでは順・逆方向とも要素が一致したが、別環境では浮動小数点の末尾が異なる可能性がある。
- SMBと遅延教師は公開実装と同じくGPUに常駐する。SMBは固定容量Tensorで保持し、EM抽出はGPU上で行う。
- 評価指標は公開実装と同じく予測・正解をCPUで連結し、NumPyで一括集計する。
- 現在のデータローダーはChicago-Tだけであり、Singapore-T、METR-LA、PEMS-BAYには未対応である。
- exp01用のChicago-T 5 seed集約は実装済みだが、GPU実験は未実行である。t検定、13ベースライン、各アブレーション、推論時間計測は未実装である。

したがって、**Chicago-Tの公開実装を読みやすく再構成し、主要な実行ロジックと固定入力のforward出力を合わせた実装である**。論文Table 2--4全体の再現には、他データセットと比較手法の実装・実行も必要になる。

## 8 現在の制約

- `OnlineConfig.learning_rate`と`weight_decay`は記録用で、現行CLIのオンラインoptimizerには反映されない。新しいoptimizerも`WarmupConfig`の値を使う。`OnlineConfig.batch_size`もDataLoaderへは渡さず、raw互換の1に固定している。既定設定では学習率とweight decayが両設定で一致する。
- 対応ローダーはChicago-Tである。METR-LA、PEMS-BAY、Singapore-Tは、同じ`TimeWindowDataset`へ`(T,N)` Tensorを渡すローダーを追加すれば使える。
- exp01では複数seedの自動集約を実装したが、図の自動生成は実行パイプラインへ含めていない。
- グラフ伝播方向と層構成は合っているが、module名が異なるため公開checkpointをそのまま`load_state_dict`で読み込めない。
- 論文の統計的有意差検定は、複数seedの実験結果が必要であり、この実装ファイルだけでは実行しない。
- `raw/`は参照専用としてimportせず、互換性は固定入力と実験指標で検証する。

## 9 推奨する読み順

1. [`dol/config.py`](dol/config.py)：既定値と受容野。
2. [`dol/layers/LSL.py`](dol/layers/LSL.py)：地点別補正。
3. [`dol/graph/supports.py`](dol/graph/supports.py)：固定グラフ。
4. [`dol/layers/graph_convolution.py`](dol/layers/graph_convolution.py)：拡散。
5. [`dol/models/dol.py`](dol/models/dol.py)：予測モデル全体。
6. [`dol/data/windows.py`](dol/data/windows.py)：時系列窓。
7. [`dol/training/warmup.py`](dol/training/warmup.py)：初期学習。
8. [`dol/online/delay.py`](dol/online/delay.py)：遅延教師。
9. [`dol/online/replay.py`](dol/online/replay.py)：SMBとEM。
10. [`dol/online/schedule.py`](dol/online/schedule.py)：Awake/Hibernate。
11. [`dol/online/adapter.py`](dol/online/adapter.py)：SUM統合。
12. [`dol/pipeline.py`](dol/pipeline.py)：全体の呼出し関係。
