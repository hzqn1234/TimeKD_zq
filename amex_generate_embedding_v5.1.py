import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from layers.Sub_CA import SCA
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings

# Get all available GPUs detected by the system
gpus = list(range(torch.cuda.device_count()))
print('Available GPUs:', gpus)

# --- AMEX DATASET KNOWLEDGE BASE ---
PREFIX_MAP = {
    'D': 'Delinquency', # 逾期
    'S': 'Spend',       # 消费
    'P': 'Payment',     # 还款
    'B': 'Balance',     # 余额
    'R': 'Risk'         # 风险
}

class UniversalMSK(nn.Module):
    """
    A universal wrapper for HuggingFace AutoModels.
    """
    def __init__(self, model_name="gpt2", device="cuda", l_layer=6):
        super(UniversalMSK, self).__init__()
        self.device = device
        
        print(f"Loading model backbone: {model_name}...")
        try:
            self.backbone = AutoModel.from_pretrained(
                model_name, 
                output_attentions=True, 
                output_hidden_states=True,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

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
            sub_modules = ['transformer', 'model', 'encoder', 'decoder', 'backbone']
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
        elif isinstance(outputs, tuple):
            return outputs[0]
        else:
            return outputs

class GenPromptEmb(nn.Module):
    def __init__(self, model_name="gpt2", num_nodes=223, seq_len=13, device='cuda', d_model=768, l_layer=6, feature_names=None, **kwargs):  
        super(GenPromptEmb, self).__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        
        print(f"Initializing GenPromptEmb with model: {model_name}")

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
        
        # Context Window
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

        # --- FEATURE METADATA ---
        self.feature_names = feature_names
        self.feature_meta = [] 
        
        if self.feature_names is not None:
            self.feature_names = [str(fn) for fn in self.feature_names]
            for name in self.feature_names:
                if name.startswith('oneHot_'):
                    try:
                        core = name[7:] 
                        last_underscore = core.rfind('_')
                        if last_underscore != -1:
                            orig_feat = core[:last_underscore] 
                            cat_val = core[last_underscore+1:] 
                            prefix = orig_feat.split('_')[0]
                            category = PREFIX_MAP.get(prefix, 'Other') # Grouping Key
                            
                            self.feature_meta.append({
                                'type': 'onehot',
                                'name': name,
                                'orig_name': orig_feat,
                                'cat_val': cat_val,
                                'semantic_cat': category
                            })
                        else:
                            self.feature_meta.append({'type': 'numeric', 'name': name, 'semantic_cat': 'Other'})
                    except:
                        self.feature_meta.append({'type': 'numeric', 'name': name, 'semantic_cat': 'Other'})
                else:
                    prefix = name.split('_')[0] if '_' in name else 'O'
                    category = PREFIX_MAP.get(prefix, 'Other') # Grouping Key
                    self.feature_meta.append({
                        'type': 'numeric',
                        'name': name,
                        'semantic_cat': category
                    })

        self.has_warned_length = False

        self.msk_module = UniversalMSK(model_name=model_name, device=self.device, l_layer=l_layer)
        
        try:
            real_d_model = self.msk_module.backbone.config.hidden_size
        except:
            real_d_model = d_model
        self.d_model = real_d_model
        
        if self.d_model != d_model:
            print(f"[WARNING] Using ACTUAL dimension ({self.d_model}) instead of arg ({d_model}).")

        if len(gpus) > 1:
            self.gpt2 = nn.DataParallel(self.msk_module, device_ids=gpus).cuda()
        else:
            self.gpt2 = self.msk_module.to(device)
        
        self.sub_ac = SCA(d_model=self.seq_len, n_heads=1, d_ff=4*self.d_model, norm='LayerNorm',
                          attn_dropout=0.1, dropout=0.1, pre_norm=True, activation="gelu",
                          res_attention=True, n_layers=1, store_attn=False).to(self.device)
        for param in self.sub_ac.parameters():
            param.requires_grad = False
        
        # --- PROMPTS ---
        # Role-Playing Intro
        intro_text = (
            "Credit Risk Expert Analysis.\n"
            "Task: Assess default probability based on monthly financial statements.\n"
            "Categories: Delinquency(D), Spend(S), Payment(P), Balance(B), Risk(R).\n"
            "Data Report:"
        )
        self.id_gt_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_hd_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        
        self.id_suffix_gt = self.tokenizer.encode("\nGround Truth Label (1=Default): ", add_special_tokens=False)
        
        # CoT Suffix
        suffix_hd_text = (
            "\nBased on the trends in Balance and Delinquency shown above, "
            "analyze the repayment behavior. Predicted Default Risk:"
        )
        self.id_suffix_hd = self.tokenizer.encode(suffix_hd_text, add_special_tokens=False)

    def _generate_mask_batch(self, input_ids):
        batch_size, seq_len = input_ids.shape
        masks = torch.ones((batch_size, seq_len), device=input_ids.device) 
        if self.tokenizer.pad_token_id is not None:
             masks[input_ids == self.tokenizer.pad_token_id] = 0
        return masks

    def generate_embeddings(self, x, y, time_ref):
        batch_size = x.shape[0]
        current_seq_len = x.shape[1] 
        
        all_gt_ids = []
        all_hd_ids = []
        
        with torch.no_grad():
            for i in range(batch_size):
                y_label_ids = self.tokenizer.encode(str(int(y[i][0])), add_special_tokens=False)
                valid_len = len(time_ref[i])

                for t in range(current_seq_len):
                    # 1. Date & Context
                    if t < valid_len:
                        date_str = f"Time: {str(time_ref[i][t])}"
                    else:
                        date_str = "Padding"
                    date_ids = self.tokenizer.encode(date_str, add_special_tokens=False)
                    
                    feats_vals = x[i, t].cpu().numpy()
                    
                    # --- [NEW] TREND CALCULATION ---
                    # Calculate delta from t-1 if possible
                    prev_vals = None
                    if t > 0 and t < valid_len:
                         prev_vals = x[i, t-1].cpu().numpy()
                    
                    # --- [NEW] HIERARCHICAL GROUPING & TEXT GENERATION ---
                    # Organize features by Category (D, S, P, B, R)
                    
                    reserved_tokens = len(self.id_gt_intro) + len(date_ids) + len(self.id_suffix_gt) + len(y_label_ids) + 150
                    available_tokens = self.max_len - reserved_tokens
                    
                    if self.feature_names is not None and len(self.feature_names) == len(feats_vals):
                        
                        # Store lines by category
                        grouped_lines = { 'Delinquency': [], 'Spend': [], 'Payment': [], 'Balance': [], 'Risk': [], 'Other': [] }
                        
                        for idx, (meta, val) in enumerate(zip(self.feature_meta, feats_vals)):
                            cat_key = meta['semantic_cat']
                            if cat_key not in grouped_lines: cat_key = 'Other'
                            
                            line_str = ""
                            
                            # A. One-Hot Handling
                            if meta['type'] == 'onehot':
                                if val > 0.5:
                                    line_str = f"{meta['orig_name']}: Type {meta['cat_val']}"
                            
                            # B. Numeric Handling with Trend
                            else:
                                val_str = f"{val:.2f}"
                                trend_str = ""
                                # Add trend info if valid and numeric
                                if prev_vals is not None:
                                    prev = prev_vals[idx]
                                    # Simple heuristic: meaningful change > 0.01
                                    diff = val - prev
                                    if abs(diff) > 0.01:
                                        direction = "↑" if diff > 0 else "↓"
                                        trend_str = f" ({direction}{abs(diff):.2f})"
                                
                                line_str = f"{meta['name']}: {val_str}{trend_str}"

                            if line_str:
                                grouped_lines[cat_key].append(line_str)
                        
                        # Assemble the final text block
                        sections = []
                        # Define preferred order
                        order = ['Delinquency', 'Balance', 'Payment', 'Spend', 'Risk', 'Other']
                        
                        for key in order:
                            lines = grouped_lines.get(key, [])
                            if lines:
                                # Section Header + Content
                                # e.g. "Delinquency: D_39: 0.12, D_40: 0.00..."
                                # Using comma separation within section to save space, but section breaks are newlines
                                section_content = ", ".join(lines)
                                sections.append(f"[{key}] {section_content}")
                        
                        join_char = "\n" 
                        vals_str = join_char.join(sections)
                        
                    else:
                        # Fallback
                        vals_str = " ".join([f"{v:.2f}" for v in feats_vals])

                    vals_ids = self.tokenizer.encode(vals_str, add_special_tokens=False)

                    # --- SAFETY TRUNCATION ---
                    if len(vals_ids) > available_tokens:
                        if not self.has_warned_length:
                            print(f"[WARNING] Prompt length ({len(vals_ids)}) exceeds budget ({available_tokens}). Truncating.")
                            self.has_warned_length = True
                        vals_ids = vals_ids[:available_tokens]

                    gt_seq = (self.id_gt_intro + date_ids + self.tokenizer.encode("\n", add_special_tokens=False) + 
                              vals_ids + self.id_suffix_gt + y_label_ids)
                    
                    hd_seq = (self.id_hd_intro + date_ids + self.tokenizer.encode("\n", add_special_tokens=False) + 
                              vals_ids + self.id_suffix_hd)

                    all_gt_ids.append(torch.tensor(gt_seq))
                    all_hd_ids.append(torch.tensor(hd_seq))

            gt_tok_ids = torch.nn.utils.rnn.pad_sequence(all_gt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
            hd_tok_ids = torch.nn.utils.rnn.pad_sequence(all_hd_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)

            gt_masks = self._generate_mask_batch(gt_tok_ids)
            hd_masks = self._generate_mask_batch(hd_tok_ids)

            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
            else:
                ctx = torch.cuda.amp.autocast()

            with ctx:
                gt_out = self.gpt2(gt_tok_ids, gt_masks)
                hd_out = self.gpt2(hd_tok_ids, hd_masks)

            if hasattr(gt_out, 'last_hidden_state'):
                 gt_hidden = gt_out.last_hidden_state
                 hd_hidden = hd_out.last_hidden_state
            elif isinstance(gt_out, tuple):
                 gt_hidden = gt_out[0]
                 hd_hidden = hd_out[0]
            else:
                 gt_hidden = gt_out
                 hd_hidden = hd_out

            gt_emb = gt_hidden.view(batch_size, current_seq_len, -1, self.d_model)[:, :, -1, :]
            hd_emb = hd_hidden.view(batch_size, current_seq_len, -1, self.d_model)[:, :, -1, :]

            gt_emb = gt_emb.permute(0, 2, 1)
            hd_emb = hd_emb.permute(0, 2, 1)

            sub_out = self.sub_ac(gt_emb, hd_emb, hd_emb)
            
            return sub_out.permute(0, 2, 1)