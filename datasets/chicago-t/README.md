# Chicago-T データセット

Chicago-T は、シカゴ市内のタクシー需要を Community Area 単位・15分単位で集計した、DOL の都市時空間予測用データセットです。

## 概要

| 項目 | 内容 |
|---|---|
| 対象 | タクシー乗車需要（pickup demand） |
| 地域 | 米国イリノイ州シカゴ市 |
| 空間単位 | 77 Community Areas |
| 地域ID | 1〜77 |
| 期間 | 2020-01-01 00:00〜2023-12-31 23:45 |
| 時間粒度 | 15分 |
| 1日 | 96ステップ |
| 1週間 | 672ステップ |
| 総時点数 | 140,256 |
| 欠損時刻 | なし |
| 予測対象 | 各地域・各15分枠の乗車件数 |

時刻列はタイムゾーン情報を持たない `pandas.Timestamp` です。配列は夏時間の切り替え時も含め、全期間を欠損や重複のない固定15分グリッドとして保持しています。

このデータは地域別の発生需要であり、個人の軌跡や地域間のODフロー行列ではありません。

## ファイル

| ファイル | 内容 | 形状 | サイズ |
|---|---|---:|---:|
| `chicago20_23.npz` | 時刻・地域・需要を格納した処理済み配列 | `(140256, 77, 5)` | 550,893,063 bytes |
| `adj_chicago.npy` | 77地域間の二値隣接行列 | `(77, 77)` | 47,560 bytes |


## `chicago20_23.npz` の構造

NPZ内のキーは `data` です。配列の軸は次の意味を持ちます。

```text
data[時刻, Community Area, 列]
```

| 列 | 配布ファイル内の値 | DOLでの利用 |
|---:|---|---|
| `0` | `pandas.Timestamp` | 日時特徴量の生成に使用 |
| `1` | Community Area ID（1〜77） | ローダーでは未使用 |
| `2` | 全レコードで `0` | 未使用 |
| `3` | 平日 `0`、土日 `1` | 未使用 |
| `4` | 15分間の乗車件数 | 予測対象として使用 |

先頭要素の例です。

```text
[Timestamp('2020-01-01 00:00:00'), 1.0, 0, 0, 1.0]
```

配列全体の乗車件数は非負整数で、検査時の基本統計は次のとおりです。

| 統計 | 値 |
|---|---:|
| 最小値 | 0 |
| 最大値 | 453 |
| 平均 | 1.7937 |
| 合計 | 19,371,347 |
| ゼロの割合 | 70.55% |

地域・時刻によって需要量が大きく異なり、ゼロが多い疎な時系列です。この強い空間的不均衡と、COVID-19期をまたぐ長期的な分布変化が、オンライン学習で重要になります。

## 地域区分

77ノードはシカゴ市が定める77の Community Areas に対応します。これは正方形メッシュではなく、行政・統計用途の不規則なポリゴンです。

`chicago20_23.npz` には地域名や境界形状は含まれません。地図表示やPOI・人口・住宅などの静的特徴を結合する場合は、City of Chicago の [Community Area boundaries](https://data.cityofchicago.org/d/cauq-8yn6) を取得し、地域IDで対応付けます。

地域番号の例は次のとおりです。

| ID | Community Area |
|---:|---|
| 1 | Rogers Park |
| 8 | Near North Side |
| 28 | Near West Side |
| 32 | Loop |
| 56 | Garfield Ridge |
| 76 | O'Hare |

## 隣接行列

`adj_chicago.npy` は要素が `0` または `1` の対称行列です。

| 項目 | 値 |
|---|---:|
| ノード数 | 77 |
| 無向エッジ数 | 197 |
| 自己ループ | なし |
| 孤立ノード | なし |
| 最小次数 | 1 |
| 最大次数 | 9 |
| 平均次数 | 5.1169 |

公開コードには、この隣接行列をどの境界・規則から生成したかの詳細な説明がありません。再生成した行列へ置き換える場合は、地域ID順、自己ループ、対称化の扱いを必ず記録してください。

## DOLローダーでの扱い

`data_provider/data_loader.py` の `Dataset_ChicagoT` は、次の処理を行います。

1. `np.load(..., allow_pickle=True)` で `data` 配列を読む。
2. 最終列 `data[..., -1]` だけを需要値として取り出す。
3. 先頭20%の全地域・全時刻から単一の平均と標準偏差を計算する。
4. 全期間を同じ平均・標準偏差で標準化する。
5. 日時から7次元のカレンダー特徴を生成する。

生成される日時特徴は、分、時、曜日、日、年内通算日、月、月の日数です。

配布ファイルはobject配列なので `allow_pickle=True` が必要です。信頼できないNPZファイルには使用しないでください。

### 時系列分割

現在の実装は時点数を `train 20% / validation 5% / test 75%` に分けます。

| 区分 | 時点数 | 期間（履歴重複を除く） |
|---|---:|---|
| Train | 28,051 | 2020-01-01 00:00〜2020-10-19 04:30 |
| Validation | 7,013 | 2020-10-19 04:45〜2020-12-31 05:45 |
| Test / Online | 105,192 | 2020-12-31 06:00〜2023-12-31 23:45 |

ValidationとTestの配列には、入力窓を作るため直前の `seq_len` ステップも重ねて読み込まれます。これは予測対象期間の重複ではなく、境界直後の予測に必要な履歴です。

既定値は次のとおりです。

```text
seq_len  = 12ステップ = 過去3時間
pred_len = 12ステップ = 将来3時間
```

## 読み込み例

```python
from pathlib import Path

import numpy as np

root = Path("datasets/chicago-t")
data = np.load(root / "chicago20_23.npz", allow_pickle=True)["data"]
adj = np.load(root / "adj_chicago.npy")

timestamps = data[:, 0, 0]
area_ids = data[0, :, 1].astype(int)
demand = data[..., -1].astype(float)

print(data.shape)          # (140256, 77, 5)
print(adj.shape)           # (77, 77)
print(timestamps[0])       # 2020-01-01 00:00:00
print(timestamps[-1])      # 2023-12-31 23:45:00
print(area_ids[:5])        # [1 2 3 4 5]
print(demand.shape)        # (140256, 77)
```

## 原データとプライバシー

DOL論文はChicago-Tの原データ源としてCity of Chicago Data Portalを示しています。対応する公開データは [Taxi Trips (2013–2023)](https://data.cityofchicago.org/d/wrvz-psew) です。公開tripデータには乗車開始時刻と乗車・降車Community Areaが含まれます。

シカゴ市はプライバシー保護のため、公開タクシーデータの時刻を15分単位に丸め、位置情報を集約しています。詳細は [How Chicago Protects Privacy in TNP and Taxi Open Data](https://data.cityofchicago.org/stories/s/82d7-i4i2) を参照してください。

配布済みの `chicago20_23.npz` はDOL著者による処理済みデータです。公開raw dataからこのNPZを完全に再生成する前処理コードは、現在のリポジトリには含まれていません。そのため、次の2つを区別してください。

- 論文再現: 配布NPZと上記ハッシュを固定して利用する。
- データ再構築: City of Chicagoのraw tripデータから集計条件を明示して別データとして作る。

## 参考資料

- [DOL論文: Distribution-Aware Online Learning for Urban Spatiotemporal Forecasting on Streaming Data](https://www.ijcai.org/proceedings/2025/372)
- [DOL公式リポジトリ](https://github.com/cwang-nus/DOL)
- [Chicago Taxi Trips (2013–2023)](https://data.cityofchicago.org/d/wrvz-psew)
- [Chicago Community Area boundaries](https://data.cityofchicago.org/d/cauq-8yn6)
- [Chicago taxi open-data privacy methodology](https://data.cityofchicago.org/stories/s/82d7-i4i2)
