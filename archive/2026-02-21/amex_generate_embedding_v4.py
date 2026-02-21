import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel
from layers.Sub_CA import SCA
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
import warnings

# Get all available GPUs detected by the system
gpus = list(range(torch.cuda.device_count()))
print('Available GPUs:', gpus)

class UniversalMSK(nn.Module):
    """
    A universal wrapper for HuggingFace AutoModels that supports layer truncation
    and gradient checkpointing across different architectures (GPT-2, Neo, Llama, Qwen, etc.).
    """
    def __init__(self, model_name="gpt2", device="cuda", l_layer=6):
        super(UniversalMSK, self).__init__()
        self.device = device
        
        print(f"Loading model: {model_name}...")
        # Load base model with output_hidden_states=True to get embeddings
        # trust_remote_code=True is needed for some newer models like Qwen
        self.backbone = AutoModel.from_pretrained(
            model_name, 
            output_attentions=True, 
            output_hidden_states=True,
            trust_remote_code=True
        )

        # [CRITICAL] Universal Layer Truncation
        # Different models store layers in different attributes:
        # GPT2/Neo -> .h
        # BERT/RoBERTa -> .encoder.layer
        # Llama/Qwen/OPT -> .layers or .model.layers
        self._truncate_layers(l_layer)

        # Freeze parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Enable Gradient Checkpointing if supported
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable() 

    def _truncate_layers(self, keep_layers):
        """
        Recursively find the ModuleList containing layers and truncate it.
        """
        layer_attr_candidates = ['h', 'layers', 'blocks', 'layer']
        
        target_module_list = None
        parent_module = None
        attr_name = None

        # 1. Search in direct children
        for name in layer_attr_candidates:
            if hasattr(self.backbone, name) and isinstance(getattr(self.backbone, name), nn.ModuleList):
                target_module_list = getattr(self.backbone, name)
                parent_module = self.backbone
                attr_name = name
                break
        
        # 2. Search deeper (e.g., model.layers for Llama, transformer.h for Neo)
        if target_module_list is None:
            # Common sub-modules wrapping layers
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
            # Replace the ModuleList with the truncated one
            setattr(parent_module, attr_name, truncated_list)
        else:
            print(f"[WARNING] Could not find layer list to truncate for {type(self.backbone)}. Using full model.")

    def forward(self, input_ids, attention_mask):
        # Universal forward pass
        # We rely on HuggingFace's standard forward signature
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        
        # Return last_hidden_state
        return outputs.last_hidden_state


class GenPromptEmb(nn.Module):
    def __init__(self, model_name="gpt2", num_nodes=223, seq_len=13, device='cuda', d_model=768, l_layer=6, feature_names=None, **kwargs):  
        super(GenPromptEmb, self).__init__()
        self.device, self.d_model, self.num_nodes, self.seq_len = device, d_model, num_nodes, seq_len
        
        # [FIX] Load AutoTokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise e
            
        # Ensure padding token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Set pad_token to eos_token: {self.tokenizer.pad_token}")
        
        # [FIX] Get accurate context limit
        if hasattr(self.tokenizer, 'model_max_length'):
            self.max_len = self.tokenizer.model_max_length
            # Handle some tokenizers returning huge int values
            if self.max_len > 100000: 
                # Heuristic defaults
                if "gpt-neo" in model_name: self.max_len = 2048
                elif "qwen" in model_name.lower(): self.max_len = 8192 # Conservative
                else: self.max_len = 1024
        else:
            self.max_len = 1024 # Fallback
            
        print(f"Model Max Context Window: {self.max_len}")

        # Store feature names
        self.feature_names = feature_names
        if self.feature_names is not None:
            self.feature_names = [str(fn) for fn in self.feature_names]
            
        self.has_warned_length = False

        # Use the Universal MSK module
        self.msk_module = UniversalMSK(model_name=model_name, device=self.device, l_layer=l_layer)
        
        # Wrap in DataParallel if multiple GPUs
        if len(gpus) > 1:
            self.gpt2 = nn.DataParallel(self.msk_module, device_ids=gpus).cuda()
        else:
            self.gpt2 = self.msk_module.to(device)
        
        # SCA Module
        # Note: If switching models, d_model might change (e.g., Qwen-0.5B is 896, not 768)
        # We need to ensure the SCA layer matches the model output dimension.
        # Ideally, we should detect d_model from config, but arguments are passed in.
        # User MUST ensure d_model arg matches the chosen model!
        print(f"Initializing SCA with d_model={d_model}. Ensure this matches {model_name}'s hidden size!")
        
        self.sub_ac = SCA(d_model=self.seq_len, n_heads=1, d_ff=4*d_model, norm='LayerNorm',
                          attn_dropout=0.1, dropout=0.1, pre_norm=True, activation="gelu",
                          res_attention=True, n_layers=1, store_attn=False).to(self.device)
        
        for param in self.sub_ac.parameters():
            param.requires_grad = False
        
        # --- PROMPTS ---
        intro_text = "Credit risk analysis. Predict default: "
        self.id_gt_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_hd_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        
        self.id_mid = self.tokenizer.encode(" Feats: ", add_special_tokens=False)
        self.id_suffix_gt = self.tokenizer.encode(". Label: ", add_special_tokens=False)
        
        suffix_hd_text = ". Risk prob:"
        self.id_suffix_hd = self.tokenizer.encode(suffix_hd_text, add_special_tokens=False)

    def _generate_mask_batch(self, input_ids):
        batch_size, seq_len = input_ids.shape
        masks = torch.ones((batch_size, seq_len), device=input_ids.device) # Default attention mask is 1s
        # If there are padding tokens, set them to 0
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
                    if t < valid_len:
                        date_str = f"Dt:{str(time_ref[i][t])}"
                    else:
                        date_str = "PAD"
                    date_ids = self.tokenizer.encode(date_str, add_special_tokens=False)
                    
                    feats_vals = x[i, t].numpy() ## already on cpu
                    # feats_vals = x[i, t].cpu().numpy()
                    
                    # --- DYNAMIC TRUNCATION LOGIC ---
                    
                    # Calculate strict budget
                    reserved_tokens = len(self.id_gt_intro) + len(date_ids) + len(self.id_mid) + len(self.id_suffix_gt) + len(y_label_ids) + 10
                    available_tokens_for_vals = self.max_len - reserved_tokens
                    
                    # 1. Check if we can use names
                    use_names = False
                    if self.feature_names is not None and len(self.feature_names) == len(feats_vals):
                        # Quick heuristic: Name+Val is approx 5-8 tokens
                        if (len(self.feature_names) * 8) < available_tokens_for_vals:
                            use_names = True
                    
                    if use_names:
                        vals_list = [f"{n}:{v:.2f}" for n, v in zip(self.feature_names, feats_vals)]
                        vals_str = " ".join(vals_list)
                        vals_ids = self.tokenizer.encode(vals_str, add_special_tokens=False)
                        
                        if len(vals_ids) > available_tokens_for_vals:
                            use_names = False 
                    
                    if not use_names:
                        # 2. Fallback to Values Only
                        vals_list = [f"{v:.2f}" for v in feats_vals]
                        vals_str = " ".join(vals_list)
                        vals_ids = self.tokenizer.encode(vals_str, add_special_tokens=False)

                    # 3. HARD TRUNCATION
                    if len(vals_ids) > available_tokens_for_vals:
                        if not self.has_warned_length:
                            print(f"[WARNING] Input truncated! {len(vals_ids)} tokens > limit {available_tokens_for_vals}. Features lost.")
                            self.has_warned_length = True
                        vals_ids = vals_ids[:available_tokens_for_vals]

                    # Assembly
                    gt_seq = (self.id_gt_intro + date_ids + self.id_mid + 
                              vals_ids + self.id_suffix_gt + y_label_ids)
                    
                    hd_seq = (self.id_hd_intro + date_ids + self.id_mid + 
                              vals_ids + self.id_suffix_hd)

                    all_gt_ids.append(torch.tensor(gt_seq))
                    all_hd_ids.append(torch.tensor(hd_seq))

            # Padding
            gt_tok_ids = torch.nn.utils.rnn.pad_sequence(all_gt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
            hd_tok_ids = torch.nn.utils.rnn.pad_sequence(all_hd_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)

            # Generate standard 2D Attention Mask (Batch*SeqLen, Tokens)
            # Note: Previously we generated 4D mask for manual forward. Now we use standard 2D.
            gt_masks = self._generate_mask_batch(gt_tok_ids)
            hd_masks = self._generate_mask_batch(hd_tok_ids)

            # [COMPATIBILITY FIX] Handle Autocast
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
            else:
                ctx = torch.cuda.amp.autocast()

            with ctx:
                gt_out = self.gpt2(gt_tok_ids, gt_masks)
                hd_out = self.gpt2(hd_tok_ids, hd_masks)

            # Extract Embeddings (Last Token)
            # Check if output is tuple or tensor (AutoModel vs GPT2Model differences)
            if isinstance(gt_out, torch.Tensor):
                # Should not happen if wrapper returns last_hidden_state, but safe check
                gt_hidden = gt_out
                hd_hidden = hd_out
            else:
                gt_hidden = gt_out
                hd_hidden = hd_out

            gt_emb = gt_hidden.view(batch_size, current_seq_len, -1, self.d_model)[:, :, -1, :]
            hd_emb = hd_hidden.view(batch_size, current_seq_len, -1, self.d_model)[:, :, -1, :]

            gt_emb = gt_emb.permute(0, 2, 1)
            hd_emb = hd_emb.permute(0, 2, 1)

            sub_out = self.sub_ac(gt_emb, hd_emb, hd_emb)
            
            return sub_out.permute(0, 2, 1)