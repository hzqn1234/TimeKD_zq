import os
# [Critical] Prevent Tokenizer deadlock & CUDA memory fragmentation
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import torch
import time
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
        self.uidxs = uidxs
        self.label_name = label_name
        self.id_name = id_name
        
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.max_len = max_len
        
        print(f"Dataset initialized with Hard Max Length: {self.max_len}")
        
        # Extract pure Numpy arrays from Pandas to prevent RAM OOM swap!
        print("Extracting pure Numpy arrays from Pandas...")
        self.time_values = df_series['S_2'].values
        
        drop_cols = [c for c in ['customer_ID', 'S_2'] if c in df_series.columns]
        self.series_values = df_series.drop(drop_cols, axis=1).values.astype(np.float32) # Force float32 to prevent COW
        
        self.y_dict = None
        if df_y is not None:
            self.y_dict = df_y[label_name].to_dict()
            
        # Manually destroy the Pandas DataFrames so workers cannot inherit them
        self.df_series = None
        self.df_y = None

        print("Pre-computing static token IDs...")
        
        intro_text = "Credit risk analysis. Predict default: "
        self.id_gt_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_hd_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_mid = self.tokenizer.encode(" Feats: ", add_special_tokens=False)
        self.id_suffix_gt = self.tokenizer.encode(". Label: ", add_special_tokens=False)
        self.id_suffix_hd = self.tokenizer.encode(". Risk prob:", add_special_tokens=False)
        
        unique_dates = np.unique(self.time_values)
        self.date_cache = {}
        for d in unique_dates:
            self.date_cache[d] = self.tokenizer.encode(f"Dt:{str(d)}", add_special_tokens=False)
        self.pad_date_id = self.tokenizer.encode("PAD", add_special_tokens=False)

        self.base_overhead = len(self.id_gt_intro) + len(self.id_mid) + len(self.id_suffix_gt) + 5
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
        
        vals_list = [f"{v:.2f}" for v in feats_vals]
        vals_str = " ".join(vals_list)
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
        
        series = self.series_values[i1:i2+1]
        time_ref = self.time_values[i1:i2+1]
        
        if len(series.shape) == 1:
            series = series.reshape((-1,) + series.shape[-1:])

        label = 0
        if self.y_dict is not None:
            label = self.y_dict.get(idx, 0)
        
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
        
        # Round up to the nearest 128 to lock CUDA memory shapes
        batch_max_len = math.ceil(batch_max_len / 128) * 128
                
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_token_len", type=int, default=4096) 

    return parser.parse_args()

def save_train_embeddings(args, train_test='train'):
    print(f'save_train_embeddings - V8 (HDF5 Contiguous & CUDA Expandable)')
    
    input_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/'
    series = pd.read_feather(f'{input_path}/df_nn_series_{train_test}.feather')
    series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_{train_test}.feather').values
    
    y = None
    if train_test == 'train':
        y = pd.read_csv(f'{input_path}/{train_test}_labels.csv')

    dynamic_feature_names = series.drop(['customer_ID', 'S_2'], axis=1).columns.tolist()
    args.num_nodes = len(dynamic_feature_names)

    total_samples = len(series_idx)
    print(f"Total samples to process: {total_samples}")

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

    emb_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/emb_04/'
    os.makedirs(emb_path, exist_ok=True)
    
    output_h5_path = os.path.join(emb_path, f"{train_test}_embeddings_all.h5")
    print(f"Saving to: {output_h5_path}")

    with h5py.File(output_h5_path, 'w') as hf:
        # [CRITICAL FIX] REMOVED chunks=chunk_shape. 
        # This forces a flat, Contiguous O(1) layout for the 21GB dataset, skipping the massive B-Tree index overhead
        emb_dset = hf.create_dataset('embeddings', 
                                     shape=(total_samples, args.input_len, args.d_model),
                                     dtype='float32')
        
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
            
            # Using non_blocking=True to prevent pipeline locking during PCI-e transfer
            gt_input_ids = data['gt_input_ids'].view(-1, l).to(args.device, non_blocking=True)
            gt_mask = data['gt_mask'].view(-1, l).to(args.device, non_blocking=True)
            gt_lens = data['gt_lens'].view(-1).to(args.device, non_blocking=True)
            
            hd_input_ids = data['hd_input_ids'].view(-1, l).to(args.device, non_blocking=True)
            hd_mask = data['hd_mask'].view(-1, l).to(args.device, non_blocking=True)
            hd_lens = data['hd_lens'].view(-1).to(args.device, non_blocking=True)

            with torch.no_grad():
                embeddings_batch = gen_prompt_emb.forward_tokenized(
                    gt_input_ids, gt_mask, gt_lens,
                    hd_input_ids, hd_mask, hd_lens
                )
            
            embeddings_np = embeddings_batch.detach().cpu().numpy()
            
            emb_dset[global_idx : global_idx + current_batch_size] = embeddings_np
            
            global_idx += current_batch_size
            
        print("Bulk writing all Customer IDs to prevent HDF5 Heap Fragmentation...")
        id_dset[:] = [str(x[2]) for x in series_idx]

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