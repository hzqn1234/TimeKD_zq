import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from layers.Sub_CA import SCA
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

# Get all available GPUs detected by the system
gpus = list(range(torch.cuda.device_count()))
print('available gpus:',gpus)

class MSK(nn.Module):
    def __init__(self, device="cuda", l_layer=6):
        super(MSK, self).__init__()
        self.device = device
        # Use eager implementation for custom mask support
        self.gpt2 = GPT2Model.from_pretrained("gpt2", attn_implementation="eager",
                                              output_attentions=True, output_hidden_states=True)
        
        self.gpt2.h = self.gpt2.h[:l_layer]
        for param in self.gpt2.h.parameters():
            param.requires_grad = False

        # Gradient checkpointing reduces VRAM usage to prevent OOM
        self.gpt2.gradient_checkpointing_enable() 

    def custom_forward(self, x_ids, calibrated_mask):
        module = self.gpt2
        input_shape = x_ids.size()
        
        position_ids = torch.arange(0, input_shape[1], dtype=torch.long, device=x_ids.device).unsqueeze(0)

        inputs_embeds = module.wte(x_ids)
        position_embeds = module.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        for block in module.h:
            outputs = block(hidden_states, attention_mask=calibrated_mask, output_attentions=True)
            hidden_states = outputs[0]
        
        hidden_states = module.ln_f(hidden_states)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_states)

    def forward(self, x_ids, calibrated_mask):
        num_heads = self.gpt2.config.n_head
        # Prepare mask for multi-head attention (Batch, 1, Seq, Seq)
        calibrated_mask = calibrated_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)

        output = self.custom_forward(x_ids=x_ids, calibrated_mask=calibrated_mask).last_hidden_state
        return output

class GenPromptEmb(nn.Module):
    def __init__(self, model_name="gpt2", num_nodes=223, device='cuda', d_model=768, l_layer=6, feature_names=None, **kwargs):  
        super(GenPromptEmb, self).__init__()
        self.device, self.d_model, self.num_nodes = device, d_model, num_nodes
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        # Core MSK module
        self.msk_module = MSK(device=self.device, l_layer=l_layer)
        
        # Wrap the WHOLE MSK in DataParallel to split (Batch * Nodes) across GPUs
        self.gpt2 = nn.DataParallel(self.msk_module, device_ids=gpus).cuda()
        
        self.sub_ac = SCA(d_model=self.num_nodes, n_heads=1, d_ff=4*d_model, norm='LayerNorm',
                          attn_dropout=0.1, dropout=0.1, pre_norm=True, activation="gelu",
                          res_attention=True, n_layers=1, store_attn=False).to(self.device)
        
        for param in self.sub_ac.parameters():
            param.requires_grad = False
        
        # --- DYNAMIC CATEGORY MAPPING ---
        prefix_map = {'D': 'Delinquency', 'S': 'Spend', 'P': 'Payment', 'B': 'Balance', 'R': 'Risk'}
        
        self.node_cat_ids = []
        for col in feature_names:
            # Extract prefix: 'P_2' -> 'P' or 'oneHot_B_30_0.0' -> 'B'
            prefix = col.split('_')[1][0] if col.startswith('oneHot') else col[0]
            cat_str = prefix_map.get(prefix, 'Risk')
            self.node_cat_ids.append(self.tokenizer.encode(cat_str, add_special_tokens=False))
        
        # --- PRE-TOKENIZATION CACHE ---
        self.id_gt_intro = self.tokenizer.encode("For a credit default risk prediction dataset, " \
                                                 "Y label is whether the customer will default; " \
                                                 "X variables are features in these categories: " \
                                                 "Delinquency, Spend, Payment, Balance, Risk. " \
                                                 "From ", add_special_tokens=False)
        self.id_hd_intro = self.tokenizer.encode("From ", add_special_tokens=False)
        self.id_to = self.tokenizer.encode(" to ", add_special_tokens=False)
        self.id_vals_prefix = self.tokenizer.encode(", the values of a ", add_special_tokens=False)
        self.id_vals_mid = self.tokenizer.encode(" variable were ", add_special_tokens=False)
        self.id_suffix_gt = self.tokenizer.encode(" every month. The value for Y label is ", add_special_tokens=False)
        self.id_suffix_hd = self.tokenizer.encode(" every month. Forecast the value for Y label.", add_special_tokens=False)

    def _generate_mask_batch(self, input_ids):
        batch_size, seq_len = input_ids.shape
        masks = torch.zeros((batch_size, seq_len, seq_len), device=self.device)
        # Note: In a pre-tokenized setup, you should use markers if your numeric data 
        # is wrapped in them. If not using < >, this logic needs adjustment.
        start_marker = self.tokenizer.encode("<", add_special_tokens=False)[0]
        end_marker = self.tokenizer.encode(">", add_special_tokens=False)[0]

        for b in range(batch_size):
            ids = input_ids[b].tolist()
            ts_indices, lang_indices, capturing = [], [], False
            for idx, tid in enumerate(ids):
                if tid == self.tokenizer.pad_token_id: continue
                if tid == start_marker: capturing = True
                if capturing: ts_indices.append(idx)
                else: lang_indices.append(idx)
                if tid == end_marker: capturing = False
            
            if ts_indices and lang_indices:
                # Optimized mask filling
                masks[b][np.ix_(lang_indices, ts_indices)] = -100.0
                masks[b][np.ix_(ts_indices, lang_indices)] = -100.0
        return masks

    def generate_embeddings(self, x, y, time_ref):
        batch_size = x.shape[0]
        all_gt_ids = []
        all_hd_ids = []
        
        # 1. Assembly of Token IDs (Bypassing full string tokenization)
        with torch.no_grad():
            for i in range(batch_size):
                # Tokenize only the dynamic parts (dates and target)
                t1_ids = self.tokenizer.encode(str(time_ref[i][0]), add_special_tokens=False)
                t2_ids = self.tokenizer.encode(str(time_ref[i][-1]), add_special_tokens=False)
                y_label_ids = self.tokenizer.encode(str(int(y[i][0])), add_special_tokens=False)
                
                nodes_data = x[i].to(torch.int).cpu().numpy().T

                for node_idx, node_timeline in enumerate(nodes_data):
                    # Retrieve the pre-tokenized category for this specific node
                    cat_ids = self.node_cat_ids[node_idx]
                    
                    vals_str = ", ".join(map(str, node_timeline))
                    vals_ids = self.tokenizer.encode(vals_str, add_special_tokens=False)

                    # Concatenate pre-computed IDs with dynamic IDs
                    
                    # GT Sequence: [Intro] [T1] [to] [T2] [, values of a] [CAT] [variable were] [VALS] [suffix_gt] [Y]
                    # Sample GT: "For a credit default risk prediction dataset, Y label is whether the customer will default; 
                    # X variables are features in these categories: Delinquency, Spend, Payment, Balance, Risk. 
                    # From 2017-03-01 to 2018-03-01, the values of a Spend variable were 0.1, 0.2... every month. 
                    # The value for Y label is 0"
                    gt_seq = (self.id_gt_intro + t1_ids + self.id_to + t2_ids + 
                            self.id_vals_prefix + cat_ids + self.id_vals_mid + 
                            vals_ids + self.id_suffix_gt + y_label_ids)
                    
                    # HD Sequence: [Intro] [T1] [to] [T2] [, values of a] [CAT] [variable were] [VALS] [suffix_hd]
                    # Sample HD: "From 2017-03-01 to 2018-03-01, the values of a Spend variable were 0.1, 0.2... every month. 
                    # Forecast the value for Y label."
                    hd_seq = (self.id_hd_intro + t1_ids + self.id_to + t2_ids + 
                            self.id_vals_prefix + cat_ids + self.id_vals_mid + 
                            vals_ids + self.id_suffix_hd)

                    all_gt_ids.append(torch.tensor(gt_seq))
                    all_hd_ids.append(torch.tensor(hd_seq))

            # 2. Padding and Tensor conversion
            gt_tok_ids = torch.nn.utils.rnn.pad_sequence(all_gt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
            hd_tok_ids = torch.nn.utils.rnn.pad_sequence(all_hd_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)

            # 3. Mask Generation
            gt_masks = self._generate_mask_batch(gt_tok_ids)
            hd_masks = self._generate_mask_batch(hd_tok_ids)

            # 4. Forward Pass with Mixed Precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                gt_out = self.gpt2(gt_tok_ids, gt_masks)
                hd_out = self.gpt2(hd_tok_ids, hd_masks)

            # 5. Extract Embeddings & Cross-Attention
            # Shape: (Batch, Nodes, Seq_Len, D_Model) -> extract last token -> permute to (Batch, D_Model, Nodes)
            gt_emb = gt_out.view(batch_size, self.num_nodes, -1, self.d_model)[:, :, -1, :].permute(0, 2, 1)
            hd_emb = hd_out.view(batch_size, self.num_nodes, -1, self.d_model)[:, :, -1, :].permute(0, 2, 1)

            sub_out = self.sub_ac(gt_emb, hd_emb, hd_emb)
            return sub_out.permute(0, 2, 1)
            # return sub_out.permute(0, 2, 1).squeeze()