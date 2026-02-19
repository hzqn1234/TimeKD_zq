import os
# Prevent Tokenizer deadlock
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import math
import torch
import h5py
import argparse
import gc
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from amex_generate_embedding import GenPromptEmb
from tqdm import tqdm

import pandas as pd
import numpy as np
from datetime import datetime


class Amex_Dataset:
    def __init__(self, series_values, time_values_int, i1_i2_array, labels_array, date_token_cache, tokenizer, max_len):
        self.series_values = series_values
        self.time_values_int = time_values_int
        self.i1_i2_array = i1_i2_array
        self.labels_array = labels_array
        self.date_token_cache = date_token_cache
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self.pad_date_id = self.tokenizer.encode("PAD", add_special_tokens=False)
        
        self.y_cache = {
            0: self.tokenizer.encode("0", add_special_tokens=False),
            1: self.tokenizer.encode("1", add_special_tokens=False)
        }
        num_feats = self.series_values.shape[1]
        zero_vals_str = " ".join([f"{0.0:.2f}"] * num_feats)
        self.zero_vals_ids = self.tokenizer.encode(zero_vals_str, add_special_tokens=False)
        
        intro_text = "Credit risk analysis. Predict default: "
        self.id_gt_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_hd_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_mid = self.tokenizer.encode(" Feats: ", add_special_tokens=False)
        self.id_suffix_gt = self.tokenizer.encode(". Label: ", add_special_tokens=False)
        self.id_suffix_hd = self.tokenizer.encode(". Risk prob:", add_special_tokens=False)
        
        self.base_overhead = len(self.id_gt_intro) + len(self.id_mid) + len(self.id_suffix_gt) + 5

    def __len__(self):
        return len(self.i1_i2_array)

    def process_single_step(self, time_int, feats_vals, y_val, valid_step):
        y_label_ids = self.y_cache.get(y_val, self.y_cache[0])
        
        if valid_step:
            date_ids = self.date_token_cache.get(int(time_int), self.pad_date_id)
            vals_list = [f"{v:.2f}" for v in feats_vals]
            vals_ids = self.tokenizer.encode(" ".join(vals_list), add_special_tokens=False)
        else:
            date_ids = self.pad_date_id
            vals_ids = self.zero_vals_ids 
        
        reserved_tokens = self.base_overhead + len(date_ids) + len(y_label_ids)
        available_tokens = self.max_len - reserved_tokens
        
        if len(vals_ids) > available_tokens:
            vals_ids = vals_ids[:available_tokens]

        gt_seq = (self.id_gt_intro + date_ids + self.id_mid + 
                  vals_ids + self.id_suffix_gt + y_label_ids)
        hd_seq = (self.id_hd_intro + date_ids + self.id_mid + 
                  vals_ids + self.id_suffix_hd)
        
        return gt_seq, hd_seq

    def __getitem__(self, index):
        i1, i2 = self.i1_i2_array[index]
        label = self.labels_array[index]
        
        series = self.series_values[i1:i2+1]
        time_ref_ints = self.time_values_int[i1:i2+1]
        
        if len(series.shape) == 1:
            series = series.reshape((-1,) + series.shape[-1:])
        
        gt_ids_list = []
        hd_ids_list = []
        
        seq_len = 13
        valid_len = len(time_ref_ints)
        
        for t in range(seq_len):
            if t < valid_len:
                gt_seq, hd_seq = self.process_single_step(time_ref_ints[t], series[t], label, True)
            else:
                gt_seq, hd_seq = self.process_single_step(None, None, label, False)
            
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
            'gt_input_ids': gt_input_ids,
            'gt_mask': gt_mask,
            'gt_lens': gt_lens,
            'hd_input_ids': hd_input_ids,
            'hd_mask': hd_mask,
            'hd_lens': hd_lens
        }

    def collate_fn(self, batch):
        batch_max_len = 0
        for sample in batch:
            batch_max_len = max(batch_max_len, sample['gt_input_ids'].shape[1], sample['hd_input_ids'].shape[1])
        
        batch_max_len = math.ceil(batch_max_len / 512) * 512
                
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_token_len", type=int, default=4096) 
    
    # [NEW] Chunking Arguments for exact OS-Level split
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--total_chunks", type=int, default=1)

    return parser.parse_args()

def save_train_embeddings(args, train_test='train'):
    print(f'save_train_embeddings - V12 (Single-GPU Chunked + Zero IPC)')
    
    input_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/'
    series = pd.read_feather(f'{input_path}/df_nn_series_{train_test}.feather')
    series_idx_full = pd.read_feather(f'{input_path}/df_nn_series_idx_{train_test}.feather').values
    
    y = None
    if train_test == 'train':
        y = pd.read_csv(f'{input_path}/{train_test}_labels.csv')

    dynamic_feature_names = series.drop(['customer_ID', 'S_2'], axis=1).columns.tolist()
    args.num_nodes = len(dynamic_feature_names)

    # --- NEW CHUNKING LOGIC ---
    total_samples_full = len(series_idx_full)
    chunk_size = math.ceil(total_samples_full / args.total_chunks)
    start_idx = args.chunk_id * chunk_size
    end_idx = min(start_idx + chunk_size, total_samples_full)
    
    series_idx = series_idx_full[start_idx:end_idx]
    total_samples = len(series_idx)
    print(f"GPU Chunk {args.chunk_id + 1}/{args.total_chunks} processing {total_samples} samples...")
    # --------------------------

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
    
    print("Mapping strings to raw numeric arrays for zero-IPC worker pipeline...")
    labels_array = np.zeros(total_samples, dtype=np.int8)
    if y is not None:
        y_dict = y.set_index('customer_ID')['target'].to_dict()
        for i, row in enumerate(series_idx):
            labels_array[i] = y_dict.get(row[2], 0)
            
    unique_dates = series['S_2'].unique()
    date_to_int = {date: i for i, date in enumerate(unique_dates)}
    time_values_int = series['S_2'].map(date_to_int).values.astype(np.int16)
    date_token_cache = {i: gen_prompt_emb.tokenizer.encode(f"Dt:{str(date)}", add_special_tokens=False) for date, i in date_to_int.items()}

    drop_cols = [c for c in ['customer_ID', 'S_2'] if c in series.columns]
    series_values = series.drop(drop_cols, axis=1).values.astype(np.float32)
    
    i1_i2_array = series_idx[:, :2].astype(np.int32)
    
    del series
    del y
    gc.collect()
    
    dataset = Amex_Dataset(
        series_values=series_values,
        time_values_int=time_values_int,
        i1_i2_array=i1_i2_array,
        labels_array=labels_array,
        date_token_cache=date_token_cache,
        tokenizer=gen_prompt_emb.tokenizer,
        max_len=effective_max_len
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
    
    output_h5_path = os.path.join(emb_path, f"{train_test}_embeddings_chunk_{args.chunk_id}.h5")
    print(f"Saving to: {output_h5_path}")

    with h5py.File(output_h5_path, 'w') as hf:
        emb_dset = hf.create_dataset('embeddings', 
                                     shape=(total_samples, args.input_len, args.d_model),
                                     dtype='float32') 
        
        dt = h5py.string_dtype(encoding='utf-8')
        id_dset = hf.create_dataset('customer_ids', 
                                    shape=(total_samples,), 
                                    dtype=dt)

        print("Starting perfectly stable GPU-bound loop...")
        global_idx = 0 
        
        bar = tqdm(dataloader)
        for data in bar:
            current_batch_size = data['gt_input_ids'].shape[0]
            
            b, s, l = data['gt_input_ids'].shape
            
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
            
            emb_dset[global_idx : global_idx + current_batch_size] = embeddings_batch.detach().cpu().numpy()
            
            global_idx += current_batch_size
            
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