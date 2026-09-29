# exp03：層別Eckart–Young–Mirsky近似による既存LSLの冗長性検証（計画・未実装）

## 1. 主目的と検証範囲

**既存DOLのLSLについて、学習済みの地点別MLP重みを少数の共有基底で近似し、再学習せずに補正関数と予測を維持できるかを検証する。** 冗長性の存在は仮説であり、結果を得る前には結論にしない。

採用する方法は、第1層と第2層をそれぞれ地点方向に並べた **77×128行列への独立したSVD・低ランク近似**。exp01のseed 42〜46の同一checkpointから元モデルと近似モデルを作り、backboneが再学習で補償する余地をなくす。

前案からの改善は以下。

- 冗長性の直接検証を主実験A〜Cとし、定期射影によるonline正則化は発展実験Dにする。
- 第1層のみ・第2層のみ・両層の介入で、32→4→32のどこに近似誤差が生じるかを測る。
- 特異値だけでなく、ReLU前後、LSL補正、残差加算後、最終予測までを測る。
- LSL無効化対照と入力特徴の有効次元を追加し、元からLSLの寄与が小さい場合を識別する。
- 時点ごとの重みの低rank性と、online更新差分の低rank性を別に測る。

この実験における「冗長性」は次の範囲を指す。

|対象|調べる行列・量|支持できる主張|
|---|---|---|
|地点間の冗長性（主対象）|各層の77×128行列|地点ごとの重みを少数の共有方向で表せる|
|各MLP内の線形写像の冗長性（補助）|地点別の4×32、32×4行列|各線形層に必要なrankが4未満か|
|中間4ユニットの利用状況（補助）|activation、ユニット別出力寄与|未使用・寄与の小さいユニットがあるか|
|online更新の冗長性|時点間の77×128の更新差分|観測した更新が少数の共有方向へ集中するか|

地点方向のrankを小さくしても、MLPの中間幅は4のままである。中間幅を1〜3へ縮小できるという主張には、ユニット削除などの別の関数介入が必要になる。DOL全体や他データでの冗長性へは直ちに一般化しない。

## 2. 実装と整合するLSLのデータフロー

参照実装は [LSL.py](../../src/dol/layers/LSL.py)、[dol.py](../../src/dol/models/dol.py)、[metrics.py](../../src/dol/evaluation/metrics.py)。

~~~text
入力系列 x（各地点・各時刻で1変数）
  → 共有1×1入力射影 → h（32次元）
  → 地点別W_d, b_d → a（4次元）
  → Dropout → ReLU → z（4次元）
  → 地点別W_u, b_u → f_n(h)（32次元の補正）
  → h + f_n(h)
  → Graph WaveNet → 予測器 → 逆標準化・非負clip
~~~

評価と既存online更新はeval modeなのでDropoutは無効。以下の式はこの条件で成立する。

$$
a_n=W_{d,n}h_n+b_{d,n},\quad
z_n=\operatorname{ReLU}(a_n),\quad
f_n(h_n)=W_{u,n}z_n+b_{u,n}.
$$

$$
W_{d,n}\in\mathbb R^{4\times32},\quad b_{d,n}\in\mathbb R^4,\qquad
W_{u,n}\in\mathbb R^{32\times4},\quad b_{u,n}\in\mathbb R^{32}.
$$

Conv1dのkernel sizeは1なので、保存テンソルの末尾のサイズ1を取り除けば上記の行列になる。1地点292個、77地点で22,484個。ReLUを間に挟むため、単純に積 $W_uW_d$ をSVDしてMLPを置き換える方法は採用しない。

### 入力の32次元に関する構造上の診断

現在の入力射影は、1入力チャネルから32チャネルへのbias付き1×1 Convである。そのため各地点・時刻の特徴は、入力射影を固定すると

$$
h=a_{\rm in}x+c_{\rm in}
$$

というアフィン直線上にある。32次元の特徴が32個の独立した入力変数を意味するわけではない。中心化特徴のrankは理想演算で高々1、非中心化特徴行列のrankは高々2となる。padding位置を含め、実際にLSLへ入る特徴で確認する。

第1層はこの入力上では

$$
W_{d,n}h+b_{d,n}
=(W_{d,n}a_{\rm in})x+(W_{d,n}c_{\rm in}+b_{d,n})
$$

にしか依存しない。したがって、重みの誤差が大きくても、入力が通る方向に作用しなければ予測は変わらない場合がある。

train特徴について入力共分散の特異値を測り、近似前後の $W_{d,n}a_{\rm in}$ と $W_{d,n}c_{\rm in}+b_{d,n}$ の差を補助診断する。これは現実装から導かれる構造的性質であり、「地点間で学習された共有パターン」の証拠とは分ける。入力射影を合成した上式の出力一致も確認する。今回はこの合成を新しいモデルとして学習する実験には広げない。

## 3. 地点方向の層別SVDと定理

各地点の重みを同じ順序でflattenする。

$$
X_\ell[n,:]=\operatorname{vec}(W_{\ell,n})^\top,\qquad
X_d,X_u\in\mathbb R^{77\times128},\quad \ell\in\{d,u\}.
$$

主解析では地点平均を保存し、地点差を分解する。

$$
\mu_\ell=\frac1{77}X_\ell^\top\mathbf1,\quad
A_\ell=X_\ell-\mathbf1\mu_\ell^\top
=U_\ell\Sigma_\ell V_\ell^\top,
$$

$$
\widehat X_\ell(r_\ell)=
\mathbf1\mu_\ell^\top+
U_{\ell,r_\ell}\Sigma_{\ell,r_\ell}V_{\ell,r_\ell}^\top.
$$

$r_\ell$ は**地点差のrank**で、上限は76。再構成した $\widehat X_\ell$ 全体のrankは高々 $r_\ell+1$ になる。非中心化SVDは補助解析とし、上限77を別に扱う。

Eckart–Young–Mirsky（エッカート・ヤング・ミルスキー）定理により、

$$
\min_{\operatorname{rank}(Y)\le r}\|A_\ell-Y\|_F^2
=\sum_{i>r}\sigma_{\ell,i}^2,\qquad
\min_{\operatorname{rank}(Y)\le r}\|A_\ell-Y\|_2
=\sigma_{\ell,r+1}.
$$

全rankでは末尾誤差を0と定義する。各層のrankを固定したとき、独立の打ち切りSVDは、**固定した地点平均を保持する表現での2層の重み二乗誤差の和**も最小にする。根拠は [Eckart & Young (1936)](https://doi.org/10.1007/BF02288367)、[Mirsky (1960)](https://doi.org/10.1093/qmath/11.1.50)。

この保証は重みの近似についてのもの。予測誤差の最小化、ノイズだけの除去、過学習の解消は保証しない。特異ベクトルは重み空間の方向、$U_r\Sigma_r$ は符号付き地点座標であり、確率的な混合比や物理的なexpert名は付けない。

### MLP構造と重み誤差の関係

biasを保持すると、第1層の誤差はReLUへ渡る入力を変え、第2層の誤差は補正の出力方向を変える。$E_d=\widehat W_d-W_d$、$E_u=\widehat W_u-W_u$ とすれば、各地点・入力で

$$
\|\widehat f(h)-f(h)\|_2
\le \|E_u\|_2\|\operatorname{ReLU}(W_dh+b_d)\|_2
+\|\widehat W_u\|_2\|E_d\|_2\|h\|_2.
$$

これはReLUの1-Lipschitz性から得られる上界。層別SVDがこの上界や予測誤差を最小にするわけではないが、層ごとの誤差と入力・hidden activationの大きさを一緒に測る理由になる。

## 4. MLPの対称性を考慮する条件

主結果は保存重みを直接分解する **as_stored** とする。補助条件 **aligned** では、ReLUの正のscale変換と4ユニットの置換を使い、同じ関数を表す重みの比較を行う。

1. 各ユニットの $[W_d[j,:],b_d[j]]$ のノルムを $s_j$ とし、これを $s_j$ で割って対応する $W_u[:,j]$ を $s_j$ 倍する。
2. ノルムが $10^{-12}$ 以下のユニットはscale変換せず、件数を記録する。
3. 地点間の対応コストを24通りの順列で評価する。down+biasとupの各二乗距離を、checkpoint全地点での各ブロック平均二乗ノルムで正規化して足す。ゼロ分母のブロックは寄与0とする。
4. 対応コスト総和が最小のmedoid地点を参照にする。同点は地点ID・順列の辞書順で固定する。
5. 同じ置換を $W_d,b_d,W_u$ に適用する。downとupを別々に整列しない。

alignedで保持するbiasは変換後のbias。低rank化時にはその値を保持する。元の数値のbiasを変換後重みへ組み合わせない。

変換前後はeval modeで、固定train特徴と固定乱数入力に対して出力の一致を確認する。float32では atol=$10^{-6}$、rtol=$10^{-5}$ を用い、最大絶対・相対誤差を保存する。通らない場合は原因を確認し、alignedの結果を採用しない。

seedごとにbackboneの32チャネルの意味も変わり得るため、異なるseedの右特異部分空間をそのまま比較しない。seed間ではenergy曲線と関数・予測誤差の再現性を比較し、部分空間の時間比較は同じseed・同じ座標系で行う。近接特異値の個別ベクトルには任意性があるため、部分空間として解釈する。

## 5. 主実験A：既存重みの圧縮可能性

入力は exp01 の seed 42,43,44,45,46 の最良warm-up checkpoint。追加学習なしで、as_stored/alignedそれぞれについて2層を解析する。

$$
E_\ell(r)=\frac{\sum_{i\le r}\sigma_{\ell,i}^2}{\sum_i\sigma_{\ell,i}^2},\quad
e_{\ell,F}(r)=\sqrt{1-E_\ell(r)},\quad
r_{\ell,q}=\min\{r:E_\ell(r)\ge q\}.
$$

全 $r=0,\ldots,76$ の特異値、累積energy、絶対・相対再構成誤差、r90/r95/r99を保存する。energy 95%でも相対Frobenius誤差は約22.4%あるため、「95%保存＝誤差5%」とは表現しない。centered energyは地点別**パラメータ差**のenergyであり、入力データの分散説明率ではない。

stable rankは $\sum_i\sigma_i^2/\sigma_1^2$、entropy effective rankは $p_i=\sigma_i^2/\sum_j\sigma_j^2$ に対する $\exp(-\sum_i p_i\log p_i)$ と定義する。$A_\ell=0$ の場合は地点差なしとしてr90/r95/r99=0、誤差=0、effective rank=0とし、energy比率はnullにする。

補助診断として、各地点の4×32・32×4行列にもSVDを行い、線形写像rank $q=1,2,3,4$ の誤差を記録する。これは地点方向rank $r$ と別の軸であり、中間幅削減の証明には使わない。

## 6. 主実験B：再学習なしの関数・予測介入

### 条件とrank

中心化地点差rankを事前固定する。

$$
\mathcal R=\{0,1,2,4,8,16,32,48,64,76\}.
$$

各seed・表現で以下を実施する。「保持」は元重みを直接使用することを意味する。

|条件|第1層|第2層|目的|
|---|---|---|---|
|Original frozen|保持|保持|同じcheckpointの比較基準|
|Down-only|rank $r_d$ へ近似|保持|第1層の冗長性|
|Up-only|保持|rank $r_u$ へ近似|第2層の冗長性|
|Both|rank $r_d$ へ近似|rank $r_u$ へ近似|両層の近似が相互に与える影響|
|Random projection|同rankのrandom部分空間で近似|同左|主要特異方向を残す効果|
|LSL bypass|補正 $f_n(h)$ を0にする|—|LSL自体の予測への寄与|

rank0は層平均重みになるが、biasは地点別なので、全地点が同じMLPになるわけではない。rank76は中心化行列の完全再構成対照。LSL bypassは同じbackboneで行い、別途学習したLSLなしモデルとは区別する。

validationではBothの10×10全組合せ、Down-only/Up-onlyの各10条件を評価する。元重みと完全再構成は数値誤差確認後に重複評価を省ける。online区間の更新なし評価では、各表現につきDown-only、Up-only、Bothの対角 $r_d=r_u$ の全候補を評価する。validationで良かったrankだけを選抜しない。

Random projectionはas_storedのBoth、$r\in\{4,8,16,32\}$、固定projection seed 0,1,2に限定する。128×76の標準正規行列をQR分解して得た $Q$ の先頭 $r$ 列を使い、

$$
\widehat X_{\ell,\mathrm{rand}}=\mathbf1\mu_\ell^\top+A_\ell Q_rQ_r^\top
$$

とする。層ごとに独立な $Q$ を用い、同一seed・層ではrank間で入れ子にする。ランダム3試行はmodel seedとは別に集計する。

### 機能評価

固定backboneの同じ入力特徴を用い、以下を記録する。層別近似は毎回元checkpointから作る。

- 第1層preactivation $a$ のRMSE、ReLU activationの不一致率
- hidden出力 $z$ のRMSE、各ユニットの発火率と出力寄与RMS
- LSL補正 $f(h)$ のRMSEとrelative RMS error
- 残差加算後 $h+f(h)$ のrelative RMS error
- 最終予測のOriginalからのRMSE、および正解に対する4指標

補正の相対誤差は $\|\widehat f-f\|_F/\|f\|_F$。分母が数値的にゼロの場合はnullとし、絶対誤差と補正RMSを報告する。LSL補正が入力に比べてどれほど大きいかも測る。ReLUの符号不一致は0近傍で増え得るので、元preactivationが $|a|\le10^{-6}$ の割合を添える。

関数診断は各splitから時系列位置を等間隔に選ぶ最大1,024窓・全77地点・全入力時刻とし、選択indexを保存する。予測の4指標はsplit全窓で評価する。関数誤差は全体平均に加えて地点別の中央値・90パーセンタイル・最大値を報告する。

主評価は現在の compute_forecast_metrics と同じく**ゼロ需要を含む全要素**を対象にする。MAE、global RMSE、sample-wise RMSE、WMAPEを同じ逆標準化・非負clipで計算し、需要によるmaskを追加しない。WMAPEはJSONでは比率、図表では%表示。clipが関数差を隠す可能性があるため、予測差はclip前・後の両方を測る。

更新なしの低rankモデルは、同じく更新なしのOriginal frozenと比較する。exp01の適応あり集計値をこの比較の基準にはしない。

## 7. 主実験C：既存online軌跡での冗長性

warm-up時点だけでなく、既存のDOLが適応した重みでも低rank性が続くかを調べる。seedごとにOriginalのonline処理を1回ずつ再実行し、週境界のLSL重みとbiasを保存する。**解析のための近似重みをOriginal軌跡へ戻さない。**

[exp01 report](../exp01/report.md)に合わせて、最良checkpointのmodel/scaler、新規AdamW（lr=0.001、weight decay=0.01、空state）、空SMBへvalidationを1回投入する初期化を使う。checkpoint内のwarm-up optimizer stateを誤って復元しない。乱数seed、SMB投入順、教師遅延12、Awake1週/Hibernate1週を合わせ、Originalの4指標・更新回数をexp01と照合する。不一致は原因を記録してから比較を進める。

保存点はonline開始前と、完了した672窓ごとの境界。snapshotには最後に処理したstep、時刻、phase、更新回数、利用可能教師の範囲を記録する。

### 状態と更新差分

各層について次を別々にSVDする。

$$
X_\ell(t),\qquad
D_\ell(t)=X_\ell(t)-X_\ell(t_{\mathrm{prev}}),\qquad
D_{\ell,0}(t)=X_\ell(t)-X_\ell(0).
$$

それぞれの地点平均成分と中心化残差のノルムを保存し、同じenergy・誤差指標を計算する。Hibernate中のゼロ更新は「rank0に集中した成功例」に数えず、no_updateとして区別する。

同じseedでは、現在の更新をwarm-up時点の基底 $V_{\ell,r}(0)$ へ射影した残差も測る。

$$
e_{\rm fixed}(t,r)=
\frac{\|D_\ell^c(t)-D_\ell^c(t)V_{\ell,r}(0)V_{\ell,r}(0)^\top\|_F}
{\|D_\ell^c(t)\|_F}.
$$

時点ごとのSVD最適誤差とこの固定基底誤差を比較する。各時点で低rankでも、必要な方向が時間とともに回転すれば、固定基底で適応できるとは限らない。

### snapshotの機能診断

各snapshotをcloneし、as_storedのDown-only/Up-only/Both対角、$r\in\{4,8,16,32,76\}$を適用する。同時点の元snapshotと同じ固定train参照窓上で関数・予測差を測る。これは軌跡上のモデルの圧縮診断であり、低rankモデルを継続更新したonline性能とは呼ばない。

同じ座標系での部分空間距離と入力分布変化の関連は補助解析とする。有効rank増加だけをconcept driftの証拠とはせず、平均移動・基底回転・入力変化・後続誤差を併記する。

## 8. 発展実験D：層別射影をonline正則化へ使う（任意）

主実験の冗長性検証を完了する条件には含めない。実施する場合は結果に応じてrankを選抜せず、as_stored、両層rank $r\in\{2,4,8,16,32\}$、seed42〜46で実施する。

Original、Frozen、Layerwise EYM、同rankのRandom projectionを比較する。AdamW stateは更新条件間で継続し、biasは通常のonline更新対象のまま。EYMは重みのみを中心化して射影する。hidden-unit整列やoptimizer resetは追加しない。

主射影時点は**Awake週の最終stepの更新・予測が完了した後**。次stepから射影後重みを使う。1週672 stepでAwake/Hibernateが交互なので、0-basedの完了step 671,2015,3359,…、すなわち開始後672,2016,3360,…窓で射影する。毎暦週の射影とは区別する。教師は既存の遅延規則で解放済みのものだけを使う。

各条件は実験Cと同じ初期状態から独立に分岐する。projection用乱数はSMB用と分離する。4指標と地点・週別誤差差、SVD追加時間、射影変位、GPU peak memoryを保存する。rank76の短い接続確認で、無射影の処理と数値的に一致することを検証する。

改善が見られても、直ちに「過学習の除去」とは結論しない。急変地点での悪化も確認し、「本条件での射影による予測改善」と報告する。密な重みを更新して定期射影するため、更新対象数は22,484個のままである。

## 9. パラメータ数と判定方法

### 保持する数値の個数

Bothを層平均と2つの低rank因子で保存する場合、

$$
P(r_d,r_u)=77(4+32)+2\cdot128+(77+128)(r_d+r_u)
=3028+205(r_d+r_u).
$$

$U_r\Sigma_r$ に特異値を吸収し、$\Sigma_r$ を重複保持しない場合の数である。これは因子表現の保持数で、厳密な多様体次元や実測GPUメモリではない。

|両層同rank $r$|因子表現の保持数|元LSL 22,484個に対する割合|
|---:|---:|---:|
|0|3,028|13.5%|
|4|4,668|20.8%|
|8|6,308|28.1%|
|16|9,588|42.6%|
|32|16,148|71.8%|
|48|22,708|101.0%|
|64|29,268|130.2%|
|76|34,188|152.1%|

同rankでは $r\le47$、異なるrankでは $r_d+r_u\le94$ が元より少ない範囲。片層のみ因子化なら、もう片層をdenseで保持するため $P_{\rm one}(r)=12756+205r$ とする。実際の評価実装がdenseに再構成する場合、これらは圧縮可能な保持数として報告し、実測メモリ削減や高速化とは区別する。

### 判定

- **重みの圧縮可能性**：r90/r95/r99と連続的な誤差曲線で報告する。特定のrank閾値だけから「強い冗長性」を断定しない。
- **機能を保つ冗長性**：Bothの保持数が元未満で、同じseedのOriginal frozenに対するMAE・global/sample RMSEの相対悪化が各1%以内、補正関数relative RMS errorが5%以内かを見る。平均と各seedの達成数・最悪差を示す。1%/5%は暫定的な研究上の許容幅で、統計的同等性の証明ではない。
- **LSLの寄与が小さいケース**：bypassでも同程度の予測なら、「基底が重要なLSL機能を保存した」という主張は弱め、補正出力の保存と最終予測の鈍感さを分けて記述する。
- **更新の冗長性**：実験Cの更新差分energy集中から、観測した軌跡内の低次元性を支持する。固定基底誤差が小さくても、係数のみを継続更新した実験なしに「E-onlyで十分」とは結論しない。
- **幅4の冗長性**：局所rankや低発火率は候補の発見。幅削減後の機能評価がない限り、4→2などの縮小成功とは言わない。

5 seed平均±標本標準偏差（ddof=1）、同一seedの対応差、地点別・horizon別の誤差を報告する。WMAPEは同じ正解配列ではMAEの定数倍なので、独立した証拠として数えない。ゼロ基準値の相対差はnullとし絶対差を示す。多rank探索の最良値だけを結論にしない。必要な時系列区間推定は、予測窓を独立標本にせず、連続した週ブロックを各条件で同時に再標本化する補助解析として条件を明記する。

## 10. 図と保存構成

MatplotlibでPNGを生成し、PDFは作らない。legendは図外、rankは実数値の位置（または明記した離散軸）、全図にseed数・条件・splitを記す。結果がない段階では図を作らない。

|図|内容|
|---|---|
|singular_spectrum_by_layer.png|層別、as_stored/alignedの特異値|
|rank_vs_energy.png|累積energy、r90/r95/r99|
|rank_vs_weight_error.png|EYM理論誤差と再構成の実測誤差|
|rank_vs_function_error.png|Down-only/Up-only/Bothの補正・activation誤差|
|rank_vs_metrics.png|横軸rank、縦軸MAE/WMAPE/sample RMSE/global RMSEの2×2図|
|rank_grid_metrics.png|横軸r_d・縦軸r_uのvalidation予測誤差|
|rank_vs_parameter_count.png|片層・両層の因子保持数と元LSL基準線|
|input_and_hidden_diagnostics.png|入力有効次元、hidden利用状況、補正RMS|
|temporal_state_and_update_rank.png|状態・週更新・累積更新のrank推移|
|fixed_basis_vs_dynamic_svd.png|warm-up固定基底の誤差と各時点の最適誤差|
|online_projection_gain.png|任意Dを実行した場合のonline利得|

以下は**実装後の予定構成**。現在の計画書に加えて、実行時に必要な成果物を生成する。

~~~text
experiments/exp03/
├── strategy.md                         # 本計画・主目的・数式・比較条件
├── config.json                         # rank、seed、表現、split、許容幅
├── git_commit.txt                      # 実行コードcommitとdirty状態
├── input_manifest.json                 # データ・checkpointのpath/SHA-256
├── run.log                             # 進捗、実行条件、失敗と再開状況
├── metrics.json                        # A/B/Cの集計、seed別差、欠測条件
├── report.md                           # 結果、冗長性の根拠、限界
├── figures/                            # 上表のPNG（各図の元数値はJSON）
├── seeds/
│   └── seed<42..46>/
│       ├── config.json                 # seedに展開した設定
│       ├── manifest.json               # checkpoint参照、診断窓index
│       ├── run.log
│       ├── structure/
│       │   ├── diagnostics.json        # 入力rank、局所SVD、hidden利用状況
│       │   └── arrays/                 # 固定特徴、対応変換など。Git対象外
│       ├── static/
│       │   ├── as_stored/
│       │   │   ├── spectra.json        # 各層のcentered/uncentered SVD
│       │   │   ├── validation.json     # rank grid、片層介入、対照
│       │   │   ├── online_frozen.json  # 更新なしの全期間評価
│       │   │   └── arrays/             # 因子・診断配列。Git対象外
│       │   └── aligned/                # 同構成＋関数同値変換の誤差
│       └── trajectory/
│           ├── metrics.json            # Originalのexp01照合・予測指標
│           ├── snapshot_index.json     # step/時刻/phase/更新回数
│           ├── spectra.json            # 状態・更新差分・固定基底誤差
│           ├── intervention.json       # cloneしたsnapshotの関数診断
│           └── arrays/                 # 週別LSL重み+bias。Git対象外
└── extensions/                         # 任意Dを実行した場合だけ作成
    └── projection/
        ├── config.json                 # rank、射影時点、乱数、初期化
        ├── metrics.json
        └── seeds/seed<seed>/
            ├── original/               # 設定一致を検証してCを参照可能
            ├── frozen/
            ├── eym/r<rank>/
            └── random/r<rank>/projection_seed<0..2>/
                ├── metrics.json
                ├── diagnostics.json
                └── arrays/             # 必要な診断配列。Git対象外
~~~

checkpointはexp01を参照し、複製しない。全予測配列の恒久保存は既定で行わず、4指標・地点別/horizon別指標と必要な誤差集計を保存する。週別LSL snapshotは22,484個×4 bytesで1回約88KiB、約157回×5 seedで約67MiB（圧縮前、メタデータ・診断特徴を除く）。各条件の全モデルcheckpointを大量に作る必要はない。

既存.gitignoreの /experiments/**/arrays/、/experiments/**/checkpoints/ を使用する。計画、集計JSON、report、PNG、run.logは追跡候補。strategy.md.orig はパッチのバックアップ用ファイルで正式成果物に含めず、自動生成しない。現時点でディスク上にあるのはstrategy.mdのみ。

## 11. 実施順序と検証

1. データ・checkpoint・コード版を固定し、全rank・5 seed・対照をconfigへ記録する。
2. 行列のflatten/復元、パラメータ数、bias保持、eval modeを確認する。
3. CPU float64でSVDを行い、EYMのtail誤差と実測誤差、rank76の完全再構成を照合する。モデルへの戻しは元dtypeで行い、丸め誤差も記録する。
4. alignedの関数同値性、入力射影合成の一致、ゼロ行列時の扱いを確認する。
5. A、Bを5 seedで実行する。近似後のfine-tuningや新しいwarm-upは行わない。
6. Cで既存online軌跡を再計測し、同じrank解析とsnapshot介入を行う。
7. 主実験の結果をreportにまとめる。Dは拡張として別に実施・報告する。

主実験A〜Cは学習済みLSLの冗長性を直接調べる計画である。結論は「どちらの層を、どのrankまで、どの入力・時点で近似でき、補正関数と予測がどれほど保たれたか」という形で記述する。
