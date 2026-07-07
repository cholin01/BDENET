import torch
import numpy as np
from sklearn.metrics import mean_absolute_error
from torch import nn

mae_loss = nn.L1Loss()


def get_mask(tensor):
    # 检测 NaN 和 Inf
    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)

    # 将 NaN 和 Inf 位置标记为 0，其余位置为 1
    mask = ~(nan_mask | inf_mask)

    return mask  # 转换为 float 以用于后续计算


def masked_mae(preds, target):
    # 获取目标的 mask，屏蔽 NaN 和 Inf
    mask = get_mask(target)

    preds_mask = preds[mask]
    target_mask = target[mask]

    # has_nan_or_inf = np.any(np.isnan(target_mask) | np.isinf(target_mask))
    # print("Array targets NaN or Inf:", has_nan_or_inf)

    mae = mae_loss(preds_mask, target_mask)

    return mae
