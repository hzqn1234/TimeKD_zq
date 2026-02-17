import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
import math
# from utils import args

class Amodel(nn.Module):
    def __init__(self,  series_dim, feature_dim, target_num, hidden_num, hidden_dim,
                        d_llm=768,
                        drop_rate=0.5, use_series_oof=False, device='cuda'
                        ):
        super(Amodel, self).__init__()
        self.device=device
        hidden_feature_dropout = 0.01
        self.use_series_oof = use_series_oof
        self.input_series_block_n1 = nn.Sequential(
                                        nn.Linear(series_dim, hidden_dim),
                                        nn.LayerNorm(hidden_dim)
                                        )
        self.input_series_block_n1_t = nn.Sequential(
                                        nn.Linear(d_llm, hidden_dim),
                                        nn.LayerNorm(hidden_dim)
                                        )
        encoder_layer = nn.TransformerEncoderLayer(
                                                    d_model         = hidden_dim, 
                                                    nhead           = 16, 
                                                    dim_feedforward = 256, 
                                                    dropout         = 0.05
                                                    # dropout         = 0.1
                                                    )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.transformer_encoder_t = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.output_block = nn.Sequential(
                                         nn.BatchNorm1d(1*hidden_dim)
                                         ,nn.Linear(1*hidden_dim, 1*hidden_dim)
                                         ,nn.Dropout(0.05)
                                         ,nn.LeakyReLU()

                                         ,nn.Linear(1*hidden_dim, 1*hidden_dim)
                                         ,nn.LeakyReLU()
                                         
                                         ,nn.Linear(1*hidden_dim, target_num)
                                         ,nn.Sigmoid()
                                         )
        self.output_block_t = nn.Sequential(
                                         nn.BatchNorm1d(1*hidden_dim)
                                         ,nn.Linear(1*hidden_dim, 1*hidden_dim)
                                         ,nn.Dropout(0.05)
                                         ,nn.LeakyReLU()

                                         ,nn.Linear(1*hidden_dim, 1*hidden_dim)
                                         ,nn.LeakyReLU()
                                         
                                         ,nn.Linear(1*hidden_dim, target_num)
                                         ,nn.Sigmoid()
                                         )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def transformer_pooling(self, transfomer_message, mask):
        node_num = mask.sum(dim=-1).detach().cpu().tolist()
        node_num_int = [int(a) for a in node_num]
        pooling_feature = []
        
        for i in range(len(node_num_int)):
            sample_feature = transfomer_message[:,i,:][node_num_int[i]-1]
            pooling_feature.append(sample_feature)
        
        return torch.stack(pooling_feature,0)

    # def forward(self, x_series,mask):
    def forward(self, data):
        x_series = data['batch_series'].to(self.device)
        mask = data['batch_mask'].to(self.device)

        ## TSF
        # print(x_series.shape)
        # torch.Size([128, 13, 223])
        if data['batch_emb_tensor'] is None:
            x1_tsf_enc = self.input_series_block_n1(x_series) # [128, 13, 223]
            x1_tsf_enc = x1_tsf_enc.permute(1, 0, 2) # [13,128,256]
            x1_tsf     = self.transformer_encoder(x1_tsf_enc) # [13,128,256]
            x1_tsf_pool = self.transformer_pooling(x1_tsf, mask) # [128,256]
            y = self.output_block(x1_tsf_pool).squeeze(1)

            x1_tsf_enc_t = None
            y_t = None
            x1_tsf_pool_t = None
        else:
            ## student
            x1_tsf_enc = self.input_series_block_n1(x_series)
            x1_tsf_enc = x1_tsf_enc.permute(1, 0, 2) # [13,128,256]
            x1_tsf     = self.transformer_encoder(x1_tsf_enc) # [13,128,256]
            x1_tsf_pool = self.transformer_pooling(x1_tsf, mask) # [128,256]
            y = self.output_block(x1_tsf_pool).squeeze(1)            

            ## teacher
            x_emb = data['batch_emb_tensor'].to(self.device)
            x1_tsf_enc_t = self.input_series_block_n1_t(x_emb) # [128, 13, 768]
            x1_tsf_enc_t = x1_tsf_enc_t.permute(1, 0, 2) # [13,128,256]
            x1_tsf_t     = self.transformer_encoder_t(x1_tsf_enc_t) # [13,128,256]
            x1_tsf_pool_t = self.transformer_pooling(x1_tsf_t, mask) # [128,256]
            y_t = self.output_block_t(x1_tsf_pool_t).squeeze(1)
        return x1_tsf_enc, x1_tsf_enc_t, y, y_t, x1_tsf_pool, x1_tsf_pool_t
        # return ts_enc, prompt_enc, ts_out, prompt_out, ts_att_pool, prompt_att_avg
        # ts_out, ts_enc, ts_att_avg
