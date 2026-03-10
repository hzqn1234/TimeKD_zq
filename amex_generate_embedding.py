import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
# [优化] 移除了 Sub_CA 的导入
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
            output_attentions=False,      
            output_hidden_states=False,   
            trust_remote_code=True,
            dtype=torch.bfloat16,         
            attn_implementation="sdpa"    
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
            
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
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
            
        # [优化] 移除了自我耗损的 Sub_CA 层初始化

    def gather_last_token(self, hidden_states, lengths):
        batch_size = hidden_states.shape[0]
        idx = (lengths - 1).view(batch_size, 1, 1).expand(batch_size, 1, hidden_states.size(2))
        return hidden_states.gather(1, idx).squeeze(1)

    # [优化] 函数签名只接收一组输入
    def forward_tokenized(self, tok_ids, masks, lens):
        """
        Forward pass with Dynamic Padding & Sequential Chunking for memory stability
        """
        batch_size_total = tok_ids.shape[0]
        
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
            ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
        else:
            ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16)

        with ctx:
            hidden_list = []
            
            chunk_size = 32
            
            for i in range(0, batch_size_total, chunk_size):
                end_idx = min(i + chunk_size, batch_size_total)
                
                # Forward Pass (Single Sequence)
                out = self.gpt2(tok_ids[i:end_idx], masks[i:end_idx])
                h = out.last_hidden_state if hasattr(out, 'last_hidden_state') else (out[0] if isinstance(out, tuple) else out)
                hidden_list.append(h)
                
            hidden = torch.cat(hidden_list, dim=0)

        # 提取最后一个有效 token
        emb_flat = self.gather_last_token(hidden, lens)

        # Reshape to (Batch, Seq_len, D_model)
        real_batch_size = batch_size_total // self.seq_len
        emb = emb_flat.view(real_batch_size, self.seq_len, self.d_model)

        # [优化] 直接返回半精度 Embedding，移除了送入 Sub_CA 的冗余操作
        return emb.half() 

    def generate_embeddings(self, x, y, time_ref):
        pass