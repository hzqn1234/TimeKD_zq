import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model
from layers.Sub_CA import SCA
from typing import Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd

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
    def __init__(self, model_name="gpt2", num_nodes=223, seq_len=13, device='cuda', d_model=768, l_layer=6, feature_names=None, **kwargs):  
        super(GenPromptEmb, self).__init__()
        self.device, self.d_model, self.num_nodes, self.seq_len = device, d_model, num_nodes, seq_len
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        
        # Core MSK module
        self.msk_module = MSK(device=self.device, l_layer=l_layer)
        
        # Wrap the WHOLE MSK in DataParallel to split (Batch * Seq_Len) across GPUs
        self.gpt2 = nn.DataParallel(self.msk_module, device_ids=gpus).cuda()
        
        # NOTE: Since we are generating month-by-month, the 'channel' dim for SCA is now seq_len
        self.sub_ac = SCA(d_model=self.seq_len, n_heads=1, d_ff=4*d_model, norm='LayerNorm',
                          attn_dropout=0.1, dropout=0.1, pre_norm=True, activation="gelu",
                          res_attention=True, n_layers=1, store_attn=False).to(self.device)
        
        for param in self.sub_ac.parameters():
            param.requires_grad = False
        
        # --- PRE-TOKENIZATION CACHE ---
        # Revised prompts for Month-by-Month generation
        self.id_gt_intro = self.tokenizer.encode("For a credit default risk prediction dataset, " \
                                                 "Y label is whether the customer will default; " \
                                                 "X variables are features. At ", add_special_tokens=False)
        
        self.id_hd_intro = self.tokenizer.encode("For a credit default risk prediction dataset, " \
                                                 "X variables are features. At ", add_special_tokens=False)
        
        self.id_mid = self.tokenizer.encode(", features were ", add_special_tokens=False)
        self.id_suffix_gt = self.tokenizer.encode(". The value for Y label is ", add_special_tokens=False)
        self.id_suffix_hd = self.tokenizer.encode(". Forecast the value for Y label.", add_special_tokens=False)

    def _generate_mask_batch(self, input_ids):
        batch_size, seq_len = input_ids.shape
        masks = torch.zeros((batch_size, seq_len, seq_len), device=self.device)
        return masks

    def generate_embeddings(self, x, y, time_ref):
        # x shape: (Batch, Seq_Len, Num_Nodes)
        batch_size = x.shape[0]
        # Ensure we use the actual seq_len from input if different from init
        current_seq_len = x.shape[1] 
        
        all_gt_ids = []
        all_hd_ids = []
        
        # 1. Assembly of Token IDs
        with torch.no_grad():
            for i in range(batch_size):
                y_label_ids = self.tokenizer.encode(str(int(y[i][0])), add_special_tokens=False)
                
                # Retrieve the actual length of the timeline for this customer
                # time_ref[i] is typically a numpy array or list of dates
                valid_len = len(time_ref[i])

                # Iterate Month by Month (Time Steps)
                for t in range(current_seq_len):
                    # Check if we are in the padding region
                    if t < valid_len:
                        date_obj = time_ref[i][t]
                        date_str = str(date_obj)
                    else:
                        # Use a placeholder for padding steps
                        date_str = "PAD"
                        
                    date_ids = self.tokenizer.encode(date_str, add_special_tokens=False)
                    
                    # All features for this month
                    # x[i, t] is (Num_Nodes,)
                    feats_vals = x[i, t].cpu().numpy()
                    vals_str = ", ".join(map(str, feats_vals))
                    vals_ids = self.tokenizer.encode(vals_str, add_special_tokens=False)

                    # GT Sequence: [Intro] [Date] [Mid] [Vals] [Suffix_GT] [Y]
                    gt_seq = (self.id_gt_intro + date_ids + self.id_mid + 
                              vals_ids + self.id_suffix_gt + y_label_ids)
                    
                    # HD Sequence: [Intro] [Date] [Mid] [Vals] [Suffix_HD]
                    hd_seq = (self.id_hd_intro + date_ids + self.id_mid + 
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
            # View as (Batch, Seq_Len, Token_Seq, D_Model) -> Extract last token -> (Batch, Seq_Len, D_Model)
            gt_emb = gt_out.view(batch_size, current_seq_len, -1, self.d_model)[:, :, -1, :]
            hd_emb = hd_out.view(batch_size, current_seq_len, -1, self.d_model)[:, :, -1, :]

            # Permute to (Batch, D_Model, Seq_Len) for SCA
            # SCA treats the last dim (Seq_Len) as the 'channel'/'feature' dim for attention
            gt_emb = gt_emb.permute(0, 2, 1)
            hd_emb = hd_emb.permute(0, 2, 1)

            sub_out = self.sub_ac(gt_emb, hd_emb, hd_emb)
            
            # Return (Batch, Seq_Len, D_Model)
            return sub_out.permute(0, 2, 1)