import os
# [关键] 防止 Tokenizer 并行导致的 CPU 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import time
import math
import h5py
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from amex_generate_embedding import GenPromptEmb
from tqdm import tqdm

import pandas as pd
import numpy as np
from datetime import datetime


class Amex_Dataset:
    def __init__(self, df_series, uidxs, tokenizer, feature_names, max_len=1024, df_y=None, label_name='target', id_name='customer_ID'):
        self.df_series = df_series
        self.df_y = df_y
        self.uidxs = uidxs
        self.label_name = label_name
        self.id_name = id_name
        
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.max_len = max_len
        
        print(f"Dataset initialized with Hard Max Length: {self.max_len}")
        print("Pre-computing static token IDs & Feature Metadata...")
        
        # --- 缓存优化 & [新增] v5 完整的角色扮演 Prompt ---
        intro_text = (
            "Credit Risk Expert Analysis.\n"
            "Task: Assess default probability based on monthly financial statements.\n"
            "Categories: Delinquency(D), Spend(S), Payment(P), Balance(B), Risk(R).\n"
            "Data Report:\n"
        )
        self.id_gt_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_hd_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        
        self.id_mid = self.tokenizer.encode("\n", add_special_tokens=False)
        self.id_suffix_gt = self.tokenizer.encode("\nGround Truth Label (1=Default): ", add_special_tokens=False)
        
        # [新增] v5 的推理引导后缀
        suffix_hd_text = (
            "\nBased on the data profile shown above, "
            "analyze the repayment behavior. Predicted Default Risk:"
        )
        self.id_suffix_hd = self.tokenizer.encode(suffix_hd_text, add_special_tokens=False)
        
        unique_dates = self.df_series['S_2'].unique()
        self.date_cache = {}
        for d in unique_dates:
            self.date_cache[d] = self.tokenizer.encode(f"Time: {str(d)}", add_special_tokens=False)
        self.pad_date_id = self.tokenizer.encode("Padding", add_special_tokens=False)

        self.base_overhead = len(self.id_gt_intro) + len(self.id_mid) + len(self.id_suffix_gt) + 5
        
        # --- v5 的特征语义化映射逻辑 ---
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
                    
        print("Dataset initialization complete.")

    def __len__(self):
        return (len(self.uidxs))

    def process_single_step(self, time_val, feats_vals, y_val, valid_step):
        if valid_step:
            date_ids = self.date_cache.get(time_val, self.pad_date_id)
        else:
            date_ids = self.pad_date_id
        
        y_label_ids = self.tokenizer.encode(str(int(y_val)), add_special_tokens=False)

        reserved_tokens = self.base_overhead + len(date_ids) + len(y_label_ids)
        available_tokens = self.max_len - reserved_tokens
        
        # --- v5 的区块化拼接逻辑 (Hierarchical Grouping) ---
        if self.feature_names is not None and len(self.feature_names) == len(feats_vals):
            grouped_lines = { 'Delinquency': [], 'Spend': [], 'Payment': [], 'Balance': [], 'Risk': [], 'Other': [] }
            
            for meta, val in zip(self.feature_meta, feats_vals):
                cat_key = meta['semantic_cat']
                if cat_key not in grouped_lines: cat_key = 'Other'
                
                line_str = ""
                if meta['type'] == 'onehot':
                    if val > 0.5:
                        line_str = f"{meta['orig_name']}: Type {meta['cat_val']}"
                else:
                    line_str = f"{meta['name']}: {val:.2f}"

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
            # 极速构建策略的回退方案
            vals_list = [f"{v:.2f}" for v in feats_vals]
            vals_str = " ".join(vals_list)
        # -----------------------------------------------------------

        vals_ids = self.tokenizer.encode(vals_str, add_special_tokens=False)
        
        if len(vals_ids) > available_tokens:
            vals_ids = vals_ids[:available_tokens]

        gt_seq = (self.id_gt_intro + date_ids + self.id_mid + 
                  vals_ids + self.id_suffix_gt + y_label_ids)
        hd_seq = (self.id_hd_intro + date_ids + self.id_mid + 
                  vals_ids + self.id_suffix_hd)
        
        return gt_seq, hd_seq

    def __getitem__(self, index):
        i1, i2, idx = self.uidxs[index]
        series = self.df_series.iloc[i1:i2+1, 1:].drop(['S_2'], axis=1).values
        time_ref = self.df_series.iloc[i1:i2+1, 1:]['S_2'].values
        
        if len(series.shape) == 1:
            series = series.reshape((-1,) + series.shape[-1:])

        label = 0
        if self.df_y is not None:
            label = self.df_y.at[idx, self.label_name]
        
        gt_ids_list = []
        hd_ids_list = []
        
        seq_len = 13
        valid_len = len(time_ref)
        
        for t in range(seq_len):
            if t < valid_len:
                gt_seq, hd_seq = self.process_single_step(time_ref[t], series[t], label, True)
            else:
                gt_seq, hd_seq = self.process_single_step(None, np.zeros(series.shape[1]), label, False)
            
            gt_ids_list.append(torch.tensor(gt_seq, dtype=torch.long))
            hd_ids_list.append(torch.tensor(hd_seq, dtype=torch.long))

        local_max = 0
        for x in gt_ids_list + hd_ids_list:
            if len(x) > local_max: local_max = len(x)
        
        pad_id = self.tokenizer.pad_token_id
        
        gt_input_ids = torch.full((seq_len, local_max), pad_id, dtype=torch.long)
        gt_mask = torch.zeros((seq_len, local_max), dtype=torch.long)
        gt_lens = torch.zeros(seq_len, dtype=torch.long)
        
        hd_input_ids = torch.full((seq_len, local_max), pad_id, dtype=torch.long)
        hd_mask = torch.zeros((seq_len, local_max), dtype=torch.long)
        hd_lens = torch.zeros(seq_len, dtype=torch.long)
        
        for i in range(seq_len):
            l = len(gt_ids_list[i])
            gt_input_ids[i, :l] = gt_ids_list[i]
            gt_mask[i, :l] = 1
            gt_lens[i] = l
            l = len(hd_ids_list[i])
            hd_input_ids[i, :l] = hd_ids_list[i]
            hd_mask[i, :l] = 1
            hd_lens[i] = l

        return {
            'idx': idx,
            'gt_input_ids': gt_input_ids,
            'gt_mask': gt_mask,
            'gt_lens': gt_lens,
            'hd_input_ids': hd_input_ids,
            'hd_mask': hd_mask,
            'hd_lens': hd_lens
        }

    def collate_fn(self, batch):
        batch_idx = np.array([sample['idx'] for sample in batch])
        
        batch_max_len = 0
        for sample in batch:
            batch_max_len = max(batch_max_len, sample['gt_input_ids'].shape[1], sample['hd_input_ids'].shape[1])
                
        pad_id = self.tokenizer.pad_token_id
        batch_size = len(batch)
        seq_len = 13
        
        final_gt_ids = torch.full((batch_size, seq_len, batch_max_len), pad_id, dtype=torch.long)
        final_gt_mask = torch.zeros((batch_size, seq_len, batch_max_len), dtype=torch.long)
        final_gt_lens = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        final_hd_ids = torch.full((batch_size, seq_len, batch_max_len), pad_id, dtype=torch.long)
        final_hd_mask = torch.zeros((batch_size, seq_len, batch_max_len), dtype=torch.long)
        final_hd_lens = torch.zeros((batch_size, seq_len), dtype=torch.long)

        for i, sample in enumerate(batch):
            l = sample['gt_input_ids'].shape[1]
            final_gt_ids[i, :, :l] = sample['gt_input_ids']
            final_gt_mask[i, :, :l] = sample['gt_mask']
            final_gt_lens[i, :] = sample['gt_lens']
            
            l = sample['hd_input_ids'].shape[1]
            final_hd_ids[i, :, :l] = sample['hd_input_ids']
            final_hd_mask[i, :, :l] = sample['hd_mask']
            final_hd_lens[i, :] = sample['hd_lens']

        return {
            'batch_idx': batch_idx,
            'gt_input_ids': final_gt_ids,
            'gt_mask': final_gt_mask,
            'gt_lens': final_gt_lens,
            'hd_input_ids': final_hd_ids,
            'hd_mask': final_hd_mask,
            'hd_lens': final_hd_lens
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
    # [优化] Batch Size 建议 64 - 128
    parser.add_argument("--batch_size", type=int, default=64)
    # [优化] 限制 Token 长度防止 OOM
    parser.add_argument("--max_token_len", type=int, default=2048) 

    # [CHUNK LOGIC]
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--total_chunks", type=int, default=1)

    return parser.parse_args()

def save_train_embeddings(args, train_test='train'):
    print(f'save_train_embeddings - V5 (Pre-allocated Huge Array + Bulletproof OS Chunking)')
    
    input_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/'
    series = pd.read_feather(f'{input_path}/df_nn_series_{train_test}.feather')
    series_idx_full = pd.read_feather(f'{input_path}/df_nn_series_idx_{train_test}.feather').values
    
    y = None
    if train_test == 'train':
        y = pd.read_csv(f'{input_path}/{train_test}_labels.csv')

    dynamic_feature_names = series.drop(['customer_ID', 'S_2'], axis=1).columns.tolist()
    args.num_nodes = len(dynamic_feature_names)

    # --- PURE OS LEVEL CHUNKING LOGIC ---
    total_samples_full = len(series_idx_full)
    
    # Use np.array_split for mathematically perfect, load-balanced chunks
    splits = np.array_split(series_idx_full, args.total_chunks)
    series_idx = splits[args.chunk_id]
    
    total_samples = len(series_idx)
    print(f"GPU Chunk {args.chunk_id + 1}/{args.total_chunks} processing {total_samples} samples out of {total_samples_full}...")
    
    # Handle zero-sample edge case gracefully
    if total_samples == 0:
        print(f"Chunk {args.chunk_id} has 0 samples. Exiting early.")
        return
    # ------------------------------------

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
        df_y=y
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

    emb_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/emb_05/'
    os.makedirs(emb_path, exist_ok=True)
    
    output_h5_path = os.path.join(emb_path, f"{train_test}_embeddings_chunk_{args.chunk_id}.h5")
    print(f"Saving to: {output_h5_path}")

    # [核心优化] 预分配大数组
    with h5py.File(output_h5_path, 'w') as hf:
        
        # Prevent HDF5 crash if total_samples < batch_size
        chunk_dim_0 = min(args.batch_size, total_samples)
        chunk_shape = (chunk_dim_0, args.input_len, args.d_model)
        
        emb_dset = hf.create_dataset('embeddings', 
                                     shape=(total_samples, args.input_len, args.d_model),
                                     dtype='float32',
                                     chunks=chunk_shape)
        
        dt = h5py.string_dtype(encoding='utf-8')
        id_dset = hf.create_dataset('customer_ids', 
                                    shape=(total_samples,), 
                                    dtype=dt)

        print("Datasets created. Starting loop...")
        
        global_idx = 0 
        
        bar = tqdm(dataloader)
        for data in bar:
            batch_ids = data['batch_idx'] 
            current_batch_size = len(batch_ids)
            
            b, s, l = data['gt_input_ids'].shape
            
            gt_input_ids = data['gt_input_ids'].view(-1, l).to(args.device)
            gt_mask = data['gt_mask'].view(-1, l).to(args.device)
            gt_lens = data['gt_lens'].view(-1).to(args.device)
            
            hd_input_ids = data['hd_input_ids'].view(-1, l).to(args.device)
            hd_mask = data['hd_mask'].view(-1, l).to(args.device)
            hd_lens = data['hd_lens'].view(-1).to(args.device)

            with torch.no_grad():
                embeddings_batch = gen_prompt_emb.forward_tokenized(
                    gt_input_ids, gt_mask, gt_lens,
                    hd_input_ids, hd_mask, hd_lens
                )
            
            embeddings_np = embeddings_batch.detach().cpu().numpy()
            
            emb_dset[global_idx : global_idx + current_batch_size] = embeddings_np
            id_dset[global_idx : global_idx + current_batch_size] = [str(x) for x in batch_ids]
            
            global_idx += current_batch_size

    print("Done.")
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