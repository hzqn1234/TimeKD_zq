import torch
import torch.nn as nn
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from model.CAI_model import Amodel

class Dual(nn.Module):
    def __init__(
        self,
        device="cuda",
        channel=768,
        num_nodes=7,
        seq_len=96,
        pred_len=96,
        dropout_n=0.1,
        d_llm=768,
        e_layer=1,
        head=8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n = dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.head = head

        # series_dim, feature_dim, target_num, hidden_num, hidden_dim
        # self.ts_model = Amodel(self.num_nodes, 16, 1, 3, 128)

        # Norm
        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
       
        # Emb
        self.nodes_to_feature = nn.Linear(self.num_nodes, self.channel).to(self.device)
        self.token_to_feature = nn.Linear(self.d_llm, self.channel).to(self.device)

        # Time Series Encoder
        self.ts_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=1,attention_dropout=self.dropout_n,output_attention=True), d_model=self.channel, n_heads=self.head),
                    d_model=self.channel,
                    d_ff=4*self.d_llm,
                    dropout=self.dropout_n,
                    activation='relu'
                ) for l in range(self.e_layer)
            ],
            norm_layer=torch.nn.LayerNorm(self.channel)
        ).to(self.device)

        # Prompt Encoder
        self.prompt_encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor=1,attention_dropout=self.dropout_n,output_attention=True), d_model=self.channel, n_heads=self.head),
                    d_model=self.channel,
                    d_ff=4*self.d_llm,
                    dropout=self.dropout_n,
                    activation='relu'
                ) for l in range(self.e_layer)
            ],
            norm_layer=torch.nn.LayerNorm(self.channel)
        ).to(self.device)

        # Projection
        self.ts_pool  = nn.Sequential(
                        nn.Linear(self.seq_len, self.pred_len, bias=True).to(self.device),
                        nn.LeakyReLU().to(self.device),
        )
        self.ts_proj2 = nn.Sequential(
                        # nn.LayerNorm(self.num_nodes).to(self.device),
                        nn.LeakyReLU().to(self.device),
                        nn.Linear(self.channel, 1, bias=True).to(self.device),
                        nn.Sigmoid()
        )

        self.prompt_pool = nn.Sequential(
                        nn.Linear(self.seq_len, self.pred_len, bias=True).to(self.device),
                        nn.LeakyReLU().to(self.device),
        )
        self.prompt_proj2 = nn.Sequential(
                        # nn.LayerNorm(self.num_nodes).to(self.device),
                        # nn.LeakyReLU().to(self.device),
                        nn.Linear(self.channel, 1, bias=True).to(self.device),
                        nn.Sigmoid()
        )

    def transformer_pooling(self, transfomer_message, mask):
        node_num = mask.sum(dim=-1).detach().cpu().tolist()
        node_num_int = [int(a) for a in node_num]
        pooling_feature = []
        
        for i in range(len(node_num_int)):
            sample_feature = transfomer_message[:,i,:][node_num_int[i]-1]
            pooling_feature.append(sample_feature)
        
        return torch.stack(pooling_feature,0)
    
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    # def forward(self, x, prompt_emb):
    def forward(self, data):
        # y = data['batch_y']
        # x = data['batch_series'].to(self.device)
        mask = data['batch_mask'].to(self.device)
        # emb_tensor = data['batch_emb_tensor'].to(self.device)

        if data['batch_emb_tensor'] is not None:
            prompt_emb = data['batch_emb_tensor'].to(self.device)

            # # ADD THIS LINE FOR DEBUGGING
            # print("Shape of prompt_emb before squeeze:", prompt_emb.shape)

            # prompt_emb = prompt_emb.float().squeeze() # B, N, E
            prompt_emb = prompt_emb.float() # B, N, E

            # # ADD THIS LINE FOR DEBUGGING
            # print("Shape of prompt_emb before TS input:", prompt_emb.shape)

            # TS Input
            # ts_data = x.float() # B L N
            ts_data = data['batch_series'].to(self.device) # B L N
            # ts_out, ts_enc, ts_att_pool = self.ts_model(data)
            # ts_enc, _, ts_out, _, ts_att_pool, _ = self.ts_model(data)
            
            # # TS Norm
            ts_norm = self.normalize_layers(ts_data, 'norm') # B L N
            # ts_norm = ts_norm.permute(0,2,1) # B N L
            # # _, _, N = ts_norm.shape

            # # TS Emb
            # ts_emb = self.length_to_feature(ts_norm) # B N L -> B N C
            ts_emb = self.nodes_to_feature(ts_norm) # B L N -> B L C
            # # ts_emb = self.enc_embedding(ts_data, x_mark)

            # # TS Encoder
            # ts_enc, ts_att = self.ts_encoder(ts_emb) # B N C
            ts_enc, ts_att = self.ts_encoder(ts_emb) # B L C
            ts_att_last = ts_att[-1]
            ts_att_avg = ts_att_last.mean(dim=0)

            # #  TS Proj
            # ts_out = self.ts_proj(ts_enc) # B N 1
            # ts_out = ts_out.permute(0,2,1) # B 1 N
            # ts_out = self.ts_proj2(ts_out).squeeze() # B 1 1

            # TS Pool
            ts_enc2 = ts_enc.permute(1,0,2) # L B C
            ts_enc_pool = self.transformer_pooling(ts_enc2, mask) # B C

            # ts_enc = ts_enc.permute(0,2,1) # B C L
            # ts_enc_pool = self.ts_pool(ts_enc) # B C 1
            # ts_enc_pool = ts_enc_pool.squeeze(2) # B C

            # TS Proj
            ts_out = self.ts_proj2(ts_enc_pool).squeeze(1) # B

            # Prompt Encoder
            # print(prompt_emb.shape)
            # prompt_emb = self.token_to_feature(prompt_emb) # B N E -> B N C
            prompt_emb = self.token_to_feature(prompt_emb) # B L E -> B L C

            # # ADD THIS LINE FOR DEBUGGING
            # print("Shape of prompt_emb before encoder:", prompt_emb.shape)

            prompt_enc, prompt_att = self.prompt_encoder(prompt_emb) # B L C
            prompt_att_last = prompt_att[-1]
            prompt_att_avg = prompt_att_last.mean(dim=0)

            #  Prompt Pool
            prompt_enc2 = prompt_enc.permute(0,2,1) # B C L
            prompt_enc_pool = self.prompt_pool(prompt_enc2) # B L 1
            prompt_enc_pool = prompt_enc_pool.squeeze(2) # B L

            #  Prompt Proj
            prompt_out = self.prompt_proj2(prompt_enc_pool).squeeze(1) # B
        
        else:
            # TS Input
            # ts_data = x.float() # B L N
            # ts_out, ts_enc, ts_att_pool = self.ts_model(data)
            # ts_enc, _, ts_out, _, ts_att_pool, _ = self.ts_model(data)
            ts_data = data['batch_series'].to(self.device) # B L N

            prompt_enc = None
            prompt_out = None
            ts_att_avg = None
            prompt_att_avg = None
            
            # # TS Norm
            # ts_norm = self.normalize_layers(ts_data, 'norm')
            # ts_norm = ts_norm.permute(0,2,1) # B N L
            ts_norm = self.normalize_layers(ts_data, 'norm') # B L N
           
            # # TS Emb
            # ts_emb = self.length_to_feature(ts_norm) # B N C
            ts_emb = self.nodes_to_feature(ts_norm) # B L N -> B L C

            # # TS Encoder
            # ts_enc,_ = self.ts_encoder(ts_emb) # B N C
            ts_enc, ts_att = self.ts_encoder(ts_emb) # B L C
            ts_att_last = ts_att[-1]
            ts_att_avg = ts_att_last.mean(dim=0)

            # TS Pool
            ts_enc2 = ts_enc.permute(1,0,2) # L B C
            ts_enc_pool = self.transformer_pooling(ts_enc2, mask) # B C

            # ts_enc_pool = self.ts_pool(ts_enc) # B C 1
            # ts_enc_pool = ts_enc_pool.squeeze(2) # B C

            # TS Proj
            ts_out = self.ts_proj2(ts_enc_pool).squeeze(1) # B

            # # Proj
            # ts_out = self.ts_proj(ts_enc) # B N 1
            # ts_out = ts_out.permute(0,2,1) # B 1 N
            # ts_out = self.ts_proj2(ts_out).squeeze() # B 1 1

        return ts_enc, prompt_enc, ts_out, prompt_out, ts_att_last, prompt_att_avg
