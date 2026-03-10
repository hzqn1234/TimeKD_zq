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
    "bce_logits": nn.BCEWithLogitsLoss(), # [FIX] 新增针对 Logits 的 BCE
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}

class KDLoss(nn.Module):
    def __init__(self, feature_loss, fcst_loss, recon_loss, att_loss, distill_loss, 
                 feature_w=0.01, fcst_w=1.0, recon_w=0.5, att_w=0.01, distill_w=1.0, temperature=5.0):
        super(KDLoss, self).__init__()
        self.fcst_w = fcst_w
        self.feature_w = feature_w
        self.recon_w = recon_w
        self.att_w = att_w
        self.distill_w = distill_w
        self.temperature = temperature

        self.feature_loss = loss_dict[feature_loss]
        self.fcst_loss = loss_dict[fcst_loss]
        self.recon_loss = loss_dict[recon_loss]
        self.att_loss = loss_dict[att_loss]
        self.distill_loss = loss_dict[distill_loss]

    def forward(self, ts_enc, prompt_enc, ts_out, prompt_out, ts_att_last, prompt_att_last, real):
    
        if real.shape[0] == 1:
            ts_out = torch.tensor([ts_out]).to(real.device)
            prompt_out = torch.tensor([prompt_out]).to(real.device)

        total_loss = 0.0

        if self.fcst_w > 0:
            total_loss += self.fcst_w * self.fcst_loss(ts_out, real)
            
        if self.recon_w > 0:
            total_loss += self.recon_w * self.recon_loss(prompt_out, real)
            
        if self.distill_w > 0:
            T = self.temperature
            # [FIX] 因为传入的已经是 Logits，直接除以温度 T 并进行 Sigmoid 转换获得软标签
            soft_teacher = torch.sigmoid(prompt_out.detach() / T)
            soft_student = torch.sigmoid(ts_out / T)
            
            total_loss += self.distill_w * self.distill_loss(soft_student, soft_teacher) * (T * T)

        if self.feature_w > 0:
            total_loss += self.feature_w * self.feature_loss(ts_enc, prompt_enc.detach())

        if self.att_w > 0 and (ts_att_last is not None) and (prompt_att_last is not None):
            total_loss += self.att_w * self.att_loss(ts_att_last, prompt_att_last.detach())

        if isinstance(total_loss, float):
            total_loss = torch.tensor(0.0, requires_grad=True).to(real.device)

        return total_loss