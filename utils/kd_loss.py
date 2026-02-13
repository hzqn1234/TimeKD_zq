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
    def __init__(self, feature_loss, fcst_loss, recon_loss, att_loss, feature_w=0.01, fcst_w=1.0, recon_w = 0.5, att_w = 0.01):
        super(KDLoss, self).__init__()
        self.fcst_w = fcst_w
        self.feature_w = feature_w
        self.recon_w = recon_w
        self.att_w = att_w

        self.feature_loss = loss_dict[feature_loss]
        self.fcst_loss = loss_dict[fcst_loss]
        self.recon_loss = loss_dict[recon_loss]
        self.att_loss = loss_dict[att_loss]

    def forward(self, ts_enc, prompt_enc, ts_out, prompt_out, ts_att_last, prompt_att_last, real):
    
        # ## debug print
        # print(f'ts_enc shape: {ts_enc.shape}')
        # print(f'prompt_enc shape: {prompt_enc.shape}')
        # print(f'ts_out shape: {ts_out.shape}')
        # print(f'prompt_out shape: {prompt_out.shape}')
        # print(f'ts_att_last shape: {ts_att_last.shape}')
        # print(f'prompt_att_last shape: {prompt_att_last.shape}')
        # print(f'real shape: {real.shape}')
        
        ## handle batch_size = 1
        if real.shape[0] ==1:
            ts_out = torch.tensor([ts_out]).to(real.device)
            prompt_out = torch.tensor([prompt_out]).to(real.device)

        feature_loss = self.feature_loss(ts_enc, prompt_enc)     
        # print(ts_out.size(),real.size())
        fcst_loss = self.fcst_loss(ts_out, real)
        # recon_loss = self.recon_loss(prompt_out, real)
        # att_loss = self.att_loss(ts_att_last, prompt_att_last)

        # print(f'shapes: feature_loss:{feature_loss.shape},fcst_loss:{fcst_loss.shape},recon_loss:{recon_loss.shape},att_loss:{att_loss.shape}')
        # print(f'feature_loss:{feature_loss},fcst_loss:{fcst_loss},recon_loss:{recon_loss},att_loss:{att_loss}')

        total_loss = self.fcst_w * fcst_loss
        # total_loss = self.fcst_w * fcst_loss + (self.feature_w * feature_loss + self.recon_w * recon_loss + self.att_w * att_loss) * 0
        # total_loss = self.fcst_w * fcst_loss + self.feature_w * feature_loss + self.recon_w * recon_loss + self.att_w * att_loss
        # total_loss = self.fcst_w * fcst_loss + self.feature_w * feature_loss + self.recon_w * recon_loss

        return total_loss