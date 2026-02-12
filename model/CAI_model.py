import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable, List
import math
# from utils import args

class Amodel(nn.Module):
    def __init__(self, series_dim, feature_dim, target_num, hidden_num, hidden_dim, drop_rate=0.5, use_series_oof=False, device='cuda'):
        super(Amodel, self).__init__()
        self.device=device
        hidden_feature_dropout = 0.01
        self.use_series_oof = use_series_oof
        self.input_series_block_n1 = nn.Sequential(
                                        nn.Linear(series_dim, hidden_dim),
                                        # nn.Linear(13, hidden_dim),
                                        nn.LayerNorm(hidden_dim)
                                        )
        # self.input_series_block_n2 = nn.Sequential(
        #                                 nn.Linear(series_dim, hidden_dim)
        #                                 ,nn.LayerNorm(hidden_dim)
        #                                 )
        # self.input_feature_block = nn.Sequential(
        #                                 nn.Linear(feature_dim, hidden_dim)
        #                                 ,nn.BatchNorm1d(hidden_dim)
        #                                 ,nn.Dropout(hidden_feature_dropout)
        #                                 ,nn.LeakyReLU()
        #                                 )
        # self.gru_series = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(
                                                    d_model         = hidden_dim, 
                                                    nhead           = 16, 
                                                    dim_feedforward = 256, 
                                                    dropout         = 0.05
                                                    # dropout         = 0.1
                                                    )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # decoder_layer = nn.TransformerDecoderLayer(
        #                                             d_model         = hidden_dim, 
        #                                             nhead           = 2, 
        #                                             dim_feedforward = 128, 
        #                                             dropout         = 0.1
        #                                             )
        # self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # self.hidden_feature_block = []
        # for h in range(hidden_num-1):
        #     self.hidden_feature_block.extend([
        #                              nn.Linear(hidden_dim, hidden_dim)
        #                              ,nn.BatchNorm1d(hidden_dim)
        #                              ,nn.Dropout(hidden_feature_dropout)
        #                              # ,nn.Dropout(drop_rate)
        #                              ,nn.LeakyReLU()
        #                              ])
        # self.hidden_feature_block = nn.Sequential(*self.hidden_feature_block)

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

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    # def batch_gru(self,series,mask):
    #     node_num = mask.sum(dim=-1).detach().cpu()
    #     pack = nn.utils.rnn.pack_padded_sequence(series, node_num, batch_first=True, enforce_sorted=False)
    #     message,hidden = self.gru_series(pack)
    #     pooling_feature = []

    #     for i,n in enumerate(node_num.numpy()):
    #         n = int(n)
    #         bi = 0

    #         si = message.unsorted_indices[i]
    #         for k in range(n):

    #             if k == n-1:
    #                 sample_feature = message.data[bi+si]
    #             bi = bi + message.batch_sizes[k]

    #         pooling_feature.append(sample_feature)
    #     return torch.stack(pooling_feature,0)

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

        x1_tsf_enc = self.input_series_block_n1(x_series)
        x1_tsf_enc = x1_tsf_enc.permute(1, 0, 2)
        x1_tsf     = self.transformer_encoder(x1_tsf_enc)
        # x1_tsf_pool = x1_tsf.mean(dim=0)
        x1_tsf_pool = self.transformer_pooling(x1_tsf, mask)
        
        # ## GRU
        # x1_gru = self.input_series_block_n2(x_series)
        # x1_gru = self.batch_gru(x1_gru,mask)
        
        ## Concat
        # x1 = torch.cat([x1_tsf,x1_gru],axis=1)
        x1 = x1_tsf_pool

        y = self.output_block(x1).squeeze(1)
        return x1_tsf_enc, None, y, None, x1_tsf_pool, None
        # return ts_enc, prompt_enc, ts_out, prompt_out, ts_att_pool, prompt_att_avg
        # ts_out, ts_enc, ts_att_avg
