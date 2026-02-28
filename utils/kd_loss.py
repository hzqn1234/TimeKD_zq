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
    # 新增了 temperature 参数，默认设为 5.0
    def __init__(self, feature_loss, fcst_loss, recon_loss, att_loss, distill_loss, 
                 feature_w=0.01, fcst_w=1.0, recon_w=0.5, att_w=0.01, distill_w=1.0, temperature=5.0):
        super(KDLoss, self).__init__()
        self.fcst_w = fcst_w
        self.feature_w = feature_w
        self.recon_w = recon_w
        self.att_w = att_w
        self.distill_w = distill_w
        self.temperature = temperature  # 蒸馏温度 T

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

        # 动态累加 Loss，权重为0的项直接跳过，既加速计算也防止Frozen计算图报错
        total_loss = 0.0

        # 1. Student 预测真实标签的 Loss
        if self.fcst_w > 0:
            total_loss += self.fcst_w * self.fcst_loss(ts_out, real)
            
        # 2. Teacher 特权特征重建/预测标签的 Loss
        if self.recon_w > 0:
            total_loss += self.recon_w * self.recon_loss(prompt_out, real)
            
        # 3. 软标签对齐：Student向Teacher学习预测概率 (方案A: 引入温度缩放)
        if self.distill_w > 0:
            T = self.temperature
            eps = 1e-7 # 防止概率绝对为0或1时，logit计算出inf/nan
            
            # 截断 Teacher 和 Student 的输出概率，限制在 (eps, 1 - eps) 之间
            prompt_out_clamp = torch.clamp(prompt_out.detach(), eps, 1.0 - eps)
            ts_out_clamp = torch.clamp(ts_out, eps, 1.0 - eps)
            
            # 将概率逆向转换为 logits -> 除以温度 T -> 重新 Sigmoid 生成平滑的软标签
            soft_teacher = torch.sigmoid(torch.logit(prompt_out_clamp) / T)
            soft_student = torch.sigmoid(torch.logit(ts_out_clamp) / T)
            
            total_loss += self.distill_w * self.distill_loss(soft_student, soft_teacher)

        # 4. 特征层蒸馏 (Feature targets): Student 特征 vs Teacher 特征
        if self.feature_w > 0:
            total_loss += self.feature_w * self.feature_loss(ts_enc, prompt_enc.detach())

        # 5. 注意力对齐：Student向Teacher学习如何分布注意力权重
        if self.att_w > 0 and (ts_att_last is not None) and (prompt_att_last is not None):
            total_loss += self.att_w * self.att_loss(ts_att_last, prompt_att_last.detach())

        # 防止如果全部权重为0时 total_loss 是浮点数而不是 tensor
        if isinstance(total_loss, float):
            total_loss = torch.tensor(0.0, requires_grad=True).to(real.device)

        return total_loss