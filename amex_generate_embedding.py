import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from layers.Sub_CA import SCA
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings

gpus = list(range(torch.cuda.device_count()))
print('Available GPUs:', gpus)

class UniversalMSK(nn.Module):
    def __init__(self, model_name="gpt2", device="cuda", l_layer=6):
        super(UniversalMSK, self).__init__()
        self.device = device
        
        print(f"Loading model: {model_name}...")
        self.backbone = AutoModel.from_pretrained(
            model_name, 
            output_attentions=True, 
            output_hidden_states=True,
            trust_remote_code=True
        )

        self._truncate_layers(l_layer)

        for param in self.backbone.parameters():
            param.requires_grad = False

        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable() 

    def _truncate_layers(self, keep_layers):
        layer_attr_candidates = ['h', 'layers', 'blocks', 'layer']
        target_module_list = None
        parent_module = None
        attr_name = None

        for name in layer_attr_candidates:
            if hasattr(self.backbone, name) and isinstance(getattr(self.backbone, name), nn.ModuleList):
                target_module_list = getattr(self.backbone, name)
                parent_module = self.backbone
                attr_name = name
                break
        
        if target_module_list is None:
            sub_modules = ['transformer', 'model', 'encoder', 'decoder']
            for sub in sub_modules:
                if hasattr(self.backbone, sub):
                    sub_mod = getattr(self.backbone, sub)
                    for name in layer_attr_candidates:
                        if hasattr(sub_mod, name) and isinstance(getattr(sub_mod, name), nn.ModuleList):
                            target_module_list = getattr(sub_mod, name)
                            parent_module = sub_mod
                            attr_name = name
                            break
                if target_module_list: break

        if target_module_list is not None:
            print(f"Found layers at '{attr_name}'. Truncating from {len(target_module_list)} to {keep_layers} layers.")
            truncated_list = target_module_list[:keep_layers]
            setattr(parent_module, attr_name, truncated_list)
        else:
            print(f"[WARNING] Could not find layer list to truncate for {type(self.backbone)}. Using full model.")

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        return outputs[0] if isinstance(outputs, tuple) else outputs


class GenPromptEmb(nn.Module):
    def __init__(self, model_name="gpt2", num_nodes=223, seq_len=13, device='cuda', d_model=768, l_layer=6, feature_names=None, **kwargs):  
        super(GenPromptEmb, self).__init__()
        self.device, self.d_model, self.num_nodes, self.seq_len = device, d_model, num_nodes, seq_len
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise e
            
        # --- [新增] v5 稳健的 Tokenizer Padding 策略 ---
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # --- [新增] v5 Llama支持与负数异常值拦截 ---
        if hasattr(self.tokenizer, 'model_max_length'):
            self.max_len = self.tokenizer.model_max_length
            if self.max_len > 100000 or self.max_len < 0: 
                if "gpt-neo" in model_name: self.max_len = 2048
                elif "qwen" in model_name.lower(): self.max_len = 8192 
                elif "llama" in model_name.lower(): self.max_len = 4096
                else: self.max_len = 1024
        else:
            self.max_len = 1024
            
        print(f"Model Max Context Window: {self.max_len}")

        self.feature_names = feature_names
        if self.feature_names is not None:
            self.feature_names = [str(fn) for fn in self.feature_names]
            
        self.msk_module = UniversalMSK(model_name=model_name, device=self.device, l_layer=l_layer)
        
        if len(gpus) > 1:
            self.gpt2 = nn.DataParallel(self.msk_module, device_ids=gpus).cuda()
        else:
            self.gpt2 = self.msk_module.to(device)
        
        self.sub_ac = SCA(d_model=self.seq_len, n_heads=1, d_ff=4*d_model, norm='LayerNorm',
                          attn_dropout=0.1, dropout=0.1, pre_norm=True, activation="gelu",
                          res_attention=True, n_layers=1, store_attn=False).to(self.device)
        
        for param in self.sub_ac.parameters():
            param.requires_grad = False

    def gather_last_token(self, hidden_states, lengths):
        # 根据真实长度提取最后一个有效 token
        batch_size = hidden_states.shape[0]
        # Lengths - 1 gives index
        idx = (lengths - 1).view(batch_size, 1, 1).expand(batch_size, 1, hidden_states.size(2))
        return hidden_states.gather(1, idx).squeeze(1)

    def forward_tokenized(self, gt_tok_ids, gt_masks, gt_lens, hd_tok_ids, hd_masks, hd_lens):
        """
        Forward pass with Dynamic Padding
        """
        batch_size_total = gt_tok_ids.shape[0]
        
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
        else:
            ctx = torch.cuda.amp.autocast()

        with ctx:
            gt_out = self.gpt2(gt_tok_ids, gt_masks)
            hd_out = self.gpt2(hd_tok_ids, hd_masks)

        if isinstance(gt_out, torch.Tensor):
            gt_hidden = gt_out
            hd_hidden = hd_out
        elif hasattr(gt_out, 'last_hidden_state'):
            gt_hidden = gt_out.last_hidden_state
            hd_hidden = hd_out.last_hidden_state
        else:
            gt_hidden = gt_out[0]
            hd_hidden = hd_out[0]

        # Use lengths to extract the correct last token (not padding)
        gt_emb_flat = self.gather_last_token(gt_hidden, gt_lens)
        hd_emb_flat = self.gather_last_token(hd_hidden, hd_lens)

        real_batch_size = batch_size_total // self.seq_len
        
        gt_emb = gt_emb_flat.view(real_batch_size, self.seq_len, self.d_model)
        hd_emb = hd_emb_flat.view(real_batch_size, self.seq_len, self.d_model)

        gt_emb = gt_emb.permute(0, 2, 1) # (B, D, S)
        hd_emb = hd_emb.permute(0, 2, 1)

        sub_out = self.sub_ac(gt_emb, hd_emb, hd_emb)
        
        return sub_out.permute(0, 2, 1) # (B, S, D)

    def generate_embeddings(self, x, y, time_ref):
        pass