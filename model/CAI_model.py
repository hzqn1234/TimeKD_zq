import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
import math

# --- 新增：包裹原生 Attention 的 Wrapper ---
class SelfAttnWrapper(nn.Module):
    def __init__(self, original_attn):
        super().__init__()
        self.original_attn = original_attn
        self.saved_attn_weights = None
        
        # 修复报错：将原生 attention 的关键属性暴露给 Wrapper
        # 让外层的 TransformerEncoderLayer 能够正常读取
        self.batch_first = getattr(original_attn, 'batch_first', False)

    def forward(self, *args, **kwargs):
        # 核心：强制 PyTorch 底层输出 attention 权重矩阵
        kwargs['need_weights'] = True
        out, weights = self.original_attn(*args, **kwargs)
        # 将权重截获并保存到类属性中
        self.saved_attn_weights = weights
        return out, weights
# ----------------------------------------

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
                                                    )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.transformer_encoder_t = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # --- 方案A 核心拦截：替换最后一层的 self_attn 为我们的 Wrapper ---
        # 对于 Student
        last_layer_s = self.transformer_encoder.layers[-1]
        last_layer_s.self_attn = SelfAttnWrapper(last_layer_s.self_attn)
        
        # 对于 Teacher
        last_layer_t = self.transformer_encoder_t.layers[-1]
        last_layer_t.self_attn = SelfAttnWrapper(last_layer_t.self_attn)
        # ----------------------------------------------------------------

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
            
            # 提取 Student Attention 矩阵
            ts_att_matrix = self.transformer_encoder.layers[-1].self_attn.saved_attn_weights
            prompt_att_matrix = None

        else:
            ## student
            x1_tsf_enc = self.input_series_block_n1(x_series)
            x1_tsf_enc = x1_tsf_enc.permute(1, 0, 2) 
            x1_tsf     = self.transformer_encoder(x1_tsf_enc) 
            x1_tsf_pool = self.transformer_pooling(x1_tsf, mask) 
            y = self.output_block(x1_tsf_pool).squeeze(1)            
            
            # 提取 Student Attention 矩阵，维度类似 [batch_size, seq_len, seq_len]
            ts_att_matrix = self.transformer_encoder.layers[-1].self_attn.saved_attn_weights

            ## teacher
            x_emb = data['batch_emb_tensor'].to(self.device)
            x1_tsf_enc_t = self.input_series_block_n1_t(x_emb) 
            x1_tsf_enc_t = x1_tsf_enc_t.permute(1, 0, 2) 
            x1_tsf_t     = self.transformer_encoder_t(x1_tsf_enc_t) 
            x1_tsf_pool_t = self.transformer_pooling(x1_tsf_t, mask) 
            y_t = self.output_block_t(x1_tsf_pool_t).squeeze(1)
            
            # 提取 Teacher Attention 矩阵
            prompt_att_matrix = self.transformer_encoder_t.layers[-1].self_attn.saved_attn_weights

        # 将原来的 ts_att_pool, prompt_att_pool 替换成了真正的 attention 矩阵！
        return x1_tsf_enc, x1_tsf_enc_t, y, y_t, ts_att_matrix, prompt_att_matrix