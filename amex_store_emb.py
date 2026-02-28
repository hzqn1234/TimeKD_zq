print("Starting amex_store_emb.py...")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Environment set: TOKENIZERS_PARALLELISM = false")

import torch
import time
import math
import h5py
import argparse
import warnings 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from amex_generate_embedding import GenPromptEmb
from tqdm import tqdm

import pandas as pd
import numpy as np
from datetime import datetime


class Amex_Dataset:
    def __init__(self, df_series, uidxs, tokenizer, feature_names, max_len=1024, df_y=None, label_name='target', id_name='customer_ID', allow_truncate=False):
        self.df_series = df_series
        self.df_y = df_y
        self.uidxs = uidxs
        self.label_name = label_name
        self.id_name = id_name
        
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.max_len = max_len
        self.allow_truncate = allow_truncate 
        
        print(f"Dataset initialized with Hard Max Length: {self.max_len}")
        print(f"Allow Truncate Strategy: {self.allow_truncate}")
        print("Pre-computing static token IDs & Feature Metadata...")
        
        intro_text = (
            "Credit Risk Expert Analysis.\n"
            "Task: Assess default probability based on monthly financial statements.\n"
            "Categories: Delinquency(D), Spend(S), Payment(P), Balance(B), Risk(R).\n"
            "Data Report:\n"
        )
        self.id_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_mid = self.tokenizer.encode("\n", add_special_tokens=False)
        
        suffix_text = (
            "\nBased on the data profile shown above, "
            "analyze the repayment behavior. Predicted Default Risk:"
        )
        self.id_suffix = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        
        unique_dates = self.df_series['S_2'].unique()
        self.date_cache = {}
        for d in unique_dates:
            self.date_cache[d] = self.tokenizer.encode(f"Time: {str(d)}", add_special_tokens=False)
        self.pad_date_id = self.tokenizer.encode("Padding", add_special_tokens=False)

        self.base_overhead = len(self.id_intro) + len(self.id_mid) + len(self.id_suffix) + 5
        
        PREFIX_MAP = {
            'D': 'Delinquency', 'S': 'Spend', 'P': 'Payment', 
            'B': 'Balance', 'R': 'Risk'
        }
        self.feature_meta = []
        if self.feature_names is not None:
            for name in self.feature_names:
                if str(name).startswith('oneHot_'):
                    try:
                        core = name[7:] 
                        last_underscore = core.rfind('_')
                        if last_underscore != -1:
                            orig_feat = core[:last_underscore] 
                            cat_val = core[last_underscore+1:] 
                            prefix = orig_feat.split('_')[0]
                            category = PREFIX_MAP.get(prefix, 'Other')
                            self.feature_meta.append({'type': 'onehot', 'name': name, 'orig_name': orig_feat, 'cat_val': cat_val, 'semantic_cat': category})
                        else:
                            self.feature_meta.append({'type': 'numeric', 'name': name, 'semantic_cat': 'Other'})
                    except:
                        self.feature_meta.append({'type': 'numeric', 'name': name, 'semantic_cat': 'Other'})
                else:
                    prefix = str(name).split('_')[0] if '_' in str(name) else 'O'
                    category = PREFIX_MAP.get(prefix, 'Other')
                    self.feature_meta.append({'type': 'numeric', 'name': name, 'semantic_cat': category})
                    
        print("Dataset initialization complete (Single-Stream Optimized Version).")

    def __len__(self):
        return (len(self.uidxs))

    def process_single_step(self, time_val, feats_vals, prev_vals, y_val, valid_step):
        if valid_step:
            date_ids = self.date_cache.get(time_val, self.pad_date_id)
        else:
            date_ids = self.pad_date_id

        reserved_tokens = self.base_overhead + len(date_ids)
        available_tokens = self.max_len - reserved_tokens
        
        if self.feature_names is not None and len(self.feature_names) == len(feats_vals):
            grouped_lines = { 'Delinquency': [], 'Spend': [], 'Payment': [], 'Balance': [], 'Risk': [], 'Other': [] }
            
            for idx, (meta, val) in enumerate(zip(self.feature_meta, feats_vals)):
                cat_key = meta['semantic_cat']
                if cat_key not in grouped_lines: cat_key = 'Other'
                
                line_str = ""
                if meta['type'] == 'onehot':
                    if val > 0.5:
                        line_str = f"{meta['orig_name']}: Type {meta['cat_val']}"
                else:
                    val_str = f"{val:.2f}"
                    trend_str = ""
                    
                    if prev_vals is not None:
                        prev = prev_vals[idx]
                        diff = val - prev
                        if abs(diff) > 0.01:
                            direction = "↑" if diff > 0 else "↓"
                            trend_str = f" ({direction}{abs(diff):.2f})"
                    
                    line_str = f"{meta['name']}: {val_str}{trend_str}"

                if line_str:
                    grouped_lines[cat_key].append(line_str)
            
            sections = []
            order = ['Delinquency', 'Balance', 'Payment', 'Spend', 'Risk', 'Other']
            for key in order:
                lines = grouped_lines.get(key, [])
                if lines:
                    section_content = ", ".join(lines)
                    sections.append(f"[{key}] {section_content}")
            
            vals_str = "\n".join(sections)
        else:
            vals_list = [f"{v:.2f}" for v in feats_vals]
            vals_str = " ".join(vals_list)

        vals_ids = self.tokenizer.encode(vals_str, add_special_tokens=False)
        
        raw_len = len(self.id_intro) + len(date_ids) + len(self.id_mid) + len(vals_ids) + len(self.id_suffix)
        
        if len(vals_ids) > available_tokens:
            if self.allow_truncate:
                warnings.warn(f"\n[Warning] Token limit exceeded! Required: {raw_len}, Max: {self.max_len}. "
                              f"Truncating features to fit. Tail information will be lost.")
                vals_ids = vals_ids[:available_tokens]
            else:
                pass

        seq = (self.id_intro + date_ids + self.id_mid + 
               vals_ids + self.id_suffix)
        
        return seq, raw_len

    def __getitem__(self, index):
        i1, i2, idx = self.uidxs[index]
        series = self.df_series.iloc[i1:i2+1, 1:].drop(['S_2'], axis=1).values
        time_ref = self.df_series.iloc[i1:i2+1, 1:]['S_2'].values
        
        if len(series.shape) == 1:
            series = series.reshape((-1,) + series.shape[-1:])

        label = 0
        if self.df_y is not None:
            label = self.df_y.at[idx, self.label_name]
        
        ids_list = []
        sample_raw_max = 0 
        
        seq_len = 13
        valid_len = len(time_ref)
        
        for t in range(seq_len):
            if t < valid_len:
                prev_vals = series[t-1] if t > 0 else None
                seq, raw_len = self.process_single_step(time_ref[t], series[t], prev_vals, label, True)
            else:
                seq, raw_len = self.process_single_step(None, np.zeros(series.shape[1]), None, label, False)
            
            if raw_len > sample_raw_max:
                sample_raw_max = raw_len
                
            ids_list.append(torch.tensor(seq, dtype=torch.long))

        local_max = 0
        for x in ids_list:
            if len(x) > local_max: local_max = len(x)
        
        pad_id = self.tokenizer.pad_token_id
        
        input_ids = torch.full((seq_len, local_max), pad_id, dtype=torch.long)
        mask = torch.zeros((seq_len, local_max), dtype=torch.long)
        lens = torch.zeros(seq_len, dtype=torch.long)
        
        for i in range(seq_len):
            l = len(ids_list[i])
            input_ids[i, :l] = ids_list[i]
            mask[i, :l] = 1
            lens[i] = l

        return {
            'idx': idx,
            'input_ids': input_ids,
            'mask': mask,
            'lens': lens,
            'sample_raw_max': sample_raw_max
        }

    def collate_fn(self, batch):
        batch_idx = np.array([sample['idx'] for sample in batch])
        
        batch_max_len = 0
        batch_raw_max = 0 
        for sample in batch:
            batch_max_len = max(batch_max_len, sample['input_ids'].shape[1])
            if sample['sample_raw_max'] > batch_raw_max:
                batch_raw_max = sample['sample_raw_max']
                
        pad_id = self.tokenizer.pad_token_id
        batch_size = len(batch)
        seq_len = 13
        
        final_ids = torch.full((batch_size, seq_len, batch_max_len), pad_id, dtype=torch.long)
        final_mask = torch.zeros((batch_size, seq_len, batch_max_len), dtype=torch.long)
        final_lens = torch.zeros((batch_size, seq_len), dtype=torch.long)

        for i, sample in enumerate(batch):
            l = sample['input_ids'].shape[1]
            final_ids[i, :, :l] = sample['input_ids']
            final_mask[i, :, :l] = sample['mask']
            final_lens[i, :] = sample['lens']

        return {
            'batch_idx': batch_idx,
            'input_ids': final_ids,
            'mask': final_mask,
            'lens': final_lens,
            'batch_raw_max': batch_raw_max 
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_nodes", type=int, default=223)
    parser.add_argument("--input_len", type=int, default=13)
    parser.add_argument("--output_len", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--l_layers", type=int, default=6)
    parser.add_argument("--data_type", type=str, default='original')
    parser.add_argument("--sampling", type=str, default='100pct')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_token_len", type=int, default=2048) 
    parser.add_argument("--allow_truncate", type=int, default=0, help="0: 不允许截断 (可能OOM), 1: 允许截断并在发生时发出警告")
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--total_chunks", type=int, default=1)
    
    # [新增] 接收外部传入的 emb_version 参数
    parser.add_argument("--emb_version", type=str, default="v7") 

    return parser.parse_args()

def save_train_embeddings(args, train_test='train'):
    print(f'save_train_embeddings - Single-Stream Optimization')
    
    input_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/'
    series = pd.read_feather(f'{input_path}/df_nn_series_{train_test}.feather')
    series_idx_full = pd.read_feather(f'{input_path}/df_nn_series_idx_{train_test}.feather').values
    
    y = None
    if train_test == 'train':
        y = pd.read_csv(f'{input_path}/{train_test}_labels.csv')

    dynamic_feature_names = series.drop(['customer_ID', 'S_2'], axis=1).columns.tolist()
    args.num_nodes = len(dynamic_feature_names)

    total_samples_full = len(series_idx_full)
    
    splits = np.array_split(series_idx_full, args.total_chunks)
    series_idx = splits[args.chunk_id]
    
    total_samples = len(series_idx)
    print(f"GPU Chunk {args.chunk_id + 1}/{args.total_chunks} processing {total_samples} samples out of {total_samples_full}...")
    
    if total_samples == 0:
        print(f"Chunk {args.chunk_id} has 0 samples. Exiting early.")
        return

    print("Initializing Model...")
    gen_prompt_emb = GenPromptEmb(
        model_name=args.model_name,
        num_nodes=args.num_nodes,
        device=args.device,
        input_len=args.input_len,
        output_len=args.output_len,
        d_model=args.d_model,
        l_layer=args.l_layers,
        feature_names=dynamic_feature_names,
    ).to(args.device)
    
    effective_max_len = min(args.max_token_len, gen_prompt_emb.max_len)
    
    dataset = Amex_Dataset(
        series, 
        series_idx, 
        tokenizer=gen_prompt_emb.tokenizer,
        feature_names=dynamic_feature_names,
        max_len=effective_max_len,
        df_y=y,
        allow_truncate=(args.allow_truncate == 1)
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False, 
        collate_fn=dataset.collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4
    )

    # [新增] 动态解析 emb_version 映射到 emb_XX 格式
    # 提取字符串中的数字，然后用 %02d 格式化，比如 'v7' -> '07'
    version_num = "".join(filter(str.isdigit, args.emb_version))
    if not version_num:
        version_num = "0" # fallback
    formatted_version = f"emb_{int(version_num):02d}"
    
    emb_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/{formatted_version}/'
    os.makedirs(emb_path, exist_ok=True)
    
    output_h5_path = os.path.join(emb_path, f"{train_test}_embeddings_chunk_{args.chunk_id}.h5")
    print(f"Saving to: {output_h5_path}")

    global_max_raw_len = 0 

    with h5py.File(output_h5_path, 'w') as hf:
        
        chunk_dim_0 = min(args.batch_size, total_samples)
        chunk_shape = (chunk_dim_0, args.input_len, args.d_model)
        
        emb_dset = hf.create_dataset('embeddings', 
                                     shape=(total_samples, args.input_len, args.d_model),
                                     dtype='float16', 
                                     chunks=chunk_shape)
        
        dt = h5py.string_dtype(encoding='utf-8')
        id_dset = hf.create_dataset('customer_ids', 
                                    shape=(total_samples,), 
                                    dtype=dt)

        print("Datasets created. Starting loop...")
        
        global_idx = 0 
        
        bar = tqdm(dataloader)
        for data in bar:
            if data['batch_raw_max'] > global_max_raw_len:
                global_max_raw_len = data['batch_raw_max']
                
            batch_ids = data['batch_idx'] 
            current_batch_size = len(batch_ids)
            
            b, s, l = data['input_ids'].shape
            
            input_ids = data['input_ids'].view(-1, l).to(args.device)
            mask = data['mask'].view(-1, l).to(args.device)
            lens = data['lens'].view(-1).to(args.device)

            with torch.no_grad():
                embeddings_batch = gen_prompt_emb.forward_tokenized(
                    input_ids, mask, lens
                )
            
            embeddings_np = embeddings_batch.detach().cpu().numpy().astype(np.float16)
            
            emb_dset[global_idx : global_idx + current_batch_size] = embeddings_np
            id_dset[global_idx : global_idx + current_batch_size] = [str(x) for x in batch_ids]
            
            global_idx += current_batch_size

    print("\n" + "="*60)
    print("🎉 Chunk Processing Done.")
    print(f"📊 Maximum observed raw token length: {global_max_raw_len}")
    print(f"⚙️ Current max_token_len limit: {effective_max_len}")
    
    if global_max_raw_len > effective_max_len:
        if args.allow_truncate == 1:
            print(f"⚠️ [WARN] Truncation OCCURRED! Some features were discarded.")
        else:
            print(f"❌ [ERROR] Token length exceeded limit and allow_truncate is 0!")
    else:
        print(f"✅ [OK] No truncation occurred. Token length is well within safe limits.")
    print("="*60 + "\n")
    return 

if __name__ == "__main__":
    t1 = datetime.now()
    print(f"Start at {t1.strftime('%Y-%m-%d %H:%M:%S')}")

    args = parse_args()
    print(args)
    save_train_embeddings(args, 'train')

    t2 = datetime.now()
    duration = t2 - t1
    print(f"Total time spent: {duration}")