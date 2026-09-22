"""DOLで使う予測損失。"""

from torch import Tensor

from ..data.scaler import GlobalStandardScaler


def original_scale_mae(
    prediction_normalized: Tensor,
    target_normalized: Tensor,
    scaler: GlobalStandardScaler,
) -> Tensor:
    """標準化を戻した需要値で平均絶対誤差を計算する。

    Chicago-Tの0は欠損値ではなく「乗車需要0」を意味するため、0をmaskしない。
    inverse transformはTensor演算だけなのでpredictionへの勾配を保持する。
    """

    if prediction_normalized.shape != target_normalized.shape:
        raise ValueError(
            "prediction and target must have the same shape, got "
            f"{tuple(prediction_normalized.shape)} and {tuple(target_normalized.shape)}"
        )
    prediction = scaler.inverse_transform(prediction_normalized)
    target = scaler.inverse_transform(target_normalized)
    return (prediction - target).abs().mean()
