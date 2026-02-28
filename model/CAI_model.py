import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
import math

class SelfAttnWrapper(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.original_attn = original_attn
        self.saved_attn_weights = None
        self.batch_first = getattr(original_attn, 'batch_first', False)

    def forward(self, *args, **kwargs):
        kwargs['need_weights'] = True
        out, weights = self.original_attn(*args, **kwargs)
        self.saved_attn_weights = weights
        return out, weights

class Amodel(nn.Module):
    def __init__(self,  series_dim, feature_dim, target_num, hidden_num, hidden_dim,
                        d_llm=768,
                        drop_rate=0.5, use_series_oof=False, device='cuda'
                        ):
        super(Amodel, self).__init__()
        self.device=device
        self.use_series_oof = use_series_oof
        
        # Student 接收原始数值特征
        self.input_series_block_n1 = nn.Sequential(
                                        nn.Linear(series_dim, hidden_dim),
                                        nn.LayerNorm(hidden_dim)
                                        )
        # Teacher 接收大模型特征
        self.input_series_block_n1_t = nn.Sequential(
                                        nn.Linear(d_llm, hidden_dim),
                                        nn.LayerNorm(hidden_dim)
                                        )
        # [新增] Teacher 接收原始数值特征 (超级专家)
        self.input_series_block_n1_t_raw = nn.Sequential(
                                        nn.Linear(series_dim, hidden_dim),
                                        nn.LayerNorm(hidden_dim)
                                        )
                                        
        encoder_layer = nn.TransformerEncoderLayer(
                                                    d_model         = hidden_dim, 
                                                    nhead           = 16, 
                                                    dim_feedforward = 256, 
                                                    dropout         = 0.05
                                                    )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.transformer_encoder_t = nn.TransformerEncoder(encoder_layer, num_layers=3)

        last_layer_s = self.transformer_encoder.layers[-1]
        last_layer_s.self_attn = SelfAttnWrapper(last_layer_s.self_attn)
        
        last_layer_t = self.transformer_encoder_t.layers[-1]
        last_layer_t.self_attn = SelfAttnWrapper(last_layer_t.self_attn)

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

    def forward(self, data):
        x_series = data['batch_series'].to(self.device)
        mask = data['batch_mask'].to(self.device)

        if data['batch_emb_tensor'] is None:
            x1_tsf_enc = self.input_series_block_n1(x_series) 
            x1_tsf_enc = x1_tsf_enc.permute(1, 0, 2) 
            x1_tsf     = self.transformer_encoder(x1_tsf_enc) 
            x1_tsf_pool = self.transformer_pooling(x1_tsf, mask) 
            y = self.output_block(x1_tsf_pool).squeeze(1)

            x1_tsf_enc_t = None
            y_t = None
            
            ts_att_matrix = self.transformer_encoder.layers[-1].self_attn.saved_attn_weights
            prompt_att_matrix = None

        else:
            ## student
            x1_tsf_enc = self.input_series_block_n1(x_series)
            x1_tsf_enc = x1_tsf_enc.permute(1, 0, 2) 
            x1_tsf     = self.transformer_encoder(x1_tsf_enc) 
            x1_tsf_pool = self.transformer_pooling(x1_tsf, mask) 
            y = self.output_block(x1_tsf_pool).squeeze(1)            
            
            ts_att_matrix = self.transformer_encoder.layers[-1].self_attn.saved_attn_weights

            ## teacher
            x_emb = data['batch_emb_tensor'].to(self.device)
            
            # [核心改动] Teacher 特征融合！同时处理文本与数值，直接相加融合
            t_raw_enc = self.input_series_block_n1_t_raw(x_series) # 提取数值特征 [B, S, H]
            t_emb_enc = self.input_series_block_n1_t(x_emb)        # 提取LLM特征 [B, S, H]
            
            x1_tsf_enc_t = t_raw_enc + t_emb_enc                   # 文本常识与精确数值的完美融合
            
            x1_tsf_enc_t = x1_tsf_enc_t.permute(1, 0, 2) 
            x1_tsf_t     = self.transformer_encoder_t(x1_tsf_enc_t) 
            x1_tsf_pool_t = self.transformer_pooling(x1_tsf_t, mask) 
            y_t = self.output_block_t(x1_tsf_pool_t).squeeze(1)
            
            prompt_att_matrix = self.transformer_encoder_t.layers[-1].self_attn.saved_attn_weights

        return x1_tsf_enc, x1_tsf_enc_t, y, y_t, ts_att_matrix, prompt_att_matrix