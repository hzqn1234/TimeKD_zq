import torch.nn as nn
from .similar_utils import *
from copy import deepcopy
from .losses import mape_loss, mase_loss, smape_loss
import torch

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "bce": nn.BCELoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}

class KDLoss(nn.Module):
    def __init__(self, feature_loss, fcst_loss, recon_loss, att_loss, distill_loss, feature_w=0.01, fcst_w=1.0, recon_w = 0.5, att_w = 0.01, distill_w=1.0):
        super(KDLoss, self).__init__()
        self.fcst_w = fcst_w
        self.feature_w = feature_w
        self.recon_w = recon_w
        self.att_w = att_w
        self.distill_w = distill_w

        self.feature_loss = loss_dict[feature_loss]
        self.fcst_loss = loss_dict[fcst_loss]
        self.recon_loss = loss_dict[recon_loss]
        self.att_loss = loss_dict[att_loss]
        self.distill_loss = loss_dict[distill_loss]

    def forward(self, ts_enc, prompt_enc, ts_out, prompt_out, ts_att_last, prompt_att_last, real):
    
        ## handle batch_size = 1
        if real.shape[0] == 1:
            ts_out = torch.tensor([ts_out]).to(real.device)
            prompt_out = torch.tensor([prompt_out]).to(real.device)

        # 预测结果与 Ground Truth 的 Loss
        fcst_loss = self.fcst_loss(ts_out, real)
        recon_loss = self.recon_loss(prompt_out, real)
        
        # 软标签对齐：Student向Teacher学习预测概率
        distill_loss = self.distill_loss(ts_out, prompt_out.detach())

        # 特征层蒸馏 (Feature targets): Student 特征 vs Teacher 特征
        feature_loss = self.feature_loss(ts_enc, prompt_enc.detach())

        # 注意力对齐：Student向Teacher学习如何分布注意力权重
        if ts_att_last is not None and prompt_att_last is not None:
            # 这里的 ts_att_last 和 prompt_att_last 是 [batch_size, 13, 13] 的矩阵
            # 切记要 detach Teacher 的 attention
            att_loss = self.att_loss(ts_att_last, prompt_att_last.detach())
        else:
            att_loss = torch.tensor(0.0).to(real.device)

        # 这里暂不引入 Feature Loss，继续单阶段联合训练
        total_loss = self.fcst_w * fcst_loss        + self.recon_w * recon_loss + \
                     self.feature_w * feature_loss  + self.att_w * att_loss + \
                     self.distill_w * distill_loss

        return total_loss