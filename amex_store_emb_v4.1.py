import torch
import os
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
        self.df_series = df_series
        self.df_y = df_y
        self.uidxs = uidxs
        self.label_name = label_name
        self.id_name = id_name
        
        self.tokenizer = tokenizer
        self.feature_names = feature_names
        self.max_len = max_len
        
        # Pre-compute static prompt parts IDs
        intro_text = "Credit risk analysis. Predict default: "
        self.id_gt_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_hd_intro = self.tokenizer.encode(intro_text, add_special_tokens=False)
        self.id_mid = self.tokenizer.encode(" Feats: ", add_special_tokens=False)
        self.id_suffix_gt = self.tokenizer.encode(". Label: ", add_special_tokens=False)
        self.id_suffix_hd = self.tokenizer.encode(". Risk prob:", add_special_tokens=False)
        
    def __len__(self):
        return (len(self.uidxs))

    def process_single_step(self, time_val, feats_vals, y_val, valid_step):
        # 1. Date
        if valid_step:
            date_str = f"Dt:{str(time_val)}"
        else:
            date_str = "PAD"
        date_ids = self.tokenizer.encode(date_str, add_special_tokens=False)
        
        # 2. Label
        y_label_ids = self.tokenizer.encode(str(int(y_val)), add_special_tokens=False)

        # 3. Features - Truncation Budget
        reserved_tokens = len(self.id_gt_intro) + len(date_ids) + len(self.id_mid) + len(self.id_suffix_gt) + len(y_label_ids)
        available_tokens = self.max_len - reserved_tokens
        
        use_names = False
        if self.feature_names is not None and len(self.feature_names) == len(feats_vals):
            if (len(self.feature_names) * 8) < available_tokens: 
                use_names = True
        
        vals_ids = []
        if use_names:
            vals_list = [f"{n}:{v:.2f}" for n, v in zip(self.feature_names, feats_vals)]
            vals_str = " ".join(vals_list)
            vals_ids = self.tokenizer.encode(vals_str, add_special_tokens=False)
            if len(vals_ids) > available_tokens:
                use_names = False 
        
        if not use_names:
            vals_list = [f"{v:.2f}" for v in feats_vals]
            vals_str = " ".join(vals_list)
            vals_ids = self.tokenizer.encode(vals_str, add_special_tokens=False)

        if len(vals_ids) > available_tokens:
            vals_ids = vals_ids[:available_tokens]

        # Assembly
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
            label = self.df_y.loc[idx, self.label_name]
        
        gt_ids_list = []
        hd_ids_list = []
        
        seq_len = 13
        valid_len = len(time_ref)
        
        for t in range(seq_len):
            if t < valid_len:
                t_val = time_ref[t]
                valid_step = True
                feats = series[t]
            else:
                t_val = "PAD"
                valid_step = False
                feats = np.zeros(series.shape[1]) 

            gt_seq, hd_seq = self.process_single_step(t_val, feats, label, valid_step)
            
            gt_ids_list.append(torch.tensor(gt_seq, dtype=torch.long))
            hd_ids_list.append(torch.tensor(hd_seq, dtype=torch.long))

        # --- Dynamic Padding Logic ---
        # Find max length in this sample
        local_max = 0
        for x in gt_ids_list + hd_ids_list:
            if len(x) > local_max: local_max = len(x)
        
        pad_id = self.tokenizer.pad_token_id
        
        # Prepare padded tensors
        # gt_input_ids: (13, local_max)
        gt_input_ids = torch.full((seq_len, local_max), pad_id, dtype=torch.long)
        gt_mask = torch.zeros((seq_len, local_max), dtype=torch.long)
        gt_lens = torch.zeros(seq_len, dtype=torch.long)
        
        hd_input_ids = torch.full((seq_len, local_max), pad_id, dtype=torch.long)
        hd_mask = torch.zeros((seq_len, local_max), dtype=torch.long)
        hd_lens = torch.zeros(seq_len, dtype=torch.long)
        
        for i in range(seq_len):
            # GT
            l = len(gt_ids_list[i])
            gt_input_ids[i, :l] = gt_ids_list[i]
            gt_mask[i, :l] = 1
            gt_lens[i] = l
            
            # HD
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
        
        # Determine global max length in this batch
        batch_max_len = 0
        for sample in batch:
            if sample['gt_input_ids'].shape[1] > batch_max_len:
                batch_max_len = sample['gt_input_ids'].shape[1]
            if sample['hd_input_ids'].shape[1] > batch_max_len:
                batch_max_len = sample['hd_input_ids'].shape[1]
                
        # Pad everything to batch_max_len
        pad_id = self.tokenizer.pad_token_id
        batch_size = len(batch)
        seq_len = 13
        
        # Initialize final tensors
        final_gt_ids = torch.full((batch_size, seq_len, batch_max_len), pad_id, dtype=torch.long)
        final_gt_mask = torch.zeros((batch_size, seq_len, batch_max_len), dtype=torch.long)
        final_gt_lens = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        final_hd_ids = torch.full((batch_size, seq_len, batch_max_len), pad_id, dtype=torch.long)
        final_hd_mask = torch.zeros((batch_size, seq_len, batch_max_len), dtype=torch.long)
        final_hd_lens = torch.zeros((batch_size, seq_len), dtype=torch.long)

        for i, sample in enumerate(batch):
            # GT
            src = sample['gt_input_ids'] # (13, LocalL)
            l = src.shape[1]
            final_gt_ids[i, :, :l] = src
            final_gt_mask[i, :, :l] = sample['gt_mask']
            final_gt_lens[i, :] = sample['gt_lens']
            
            # HD
            src = sample['hd_input_ids']
            l = src.shape[1]
            final_hd_ids[i, :, :l] = src
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
    parser.add_argument("--batch_size", type=int, default=1)

    return parser.parse_args()

def save_train_embeddings(args, train_test='train'):
    print(f'save_train_embeddings - Optimized Dynamic Padding')

    input_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/'
    series = pd.read_feather(f'{input_path}/df_nn_series_{train_test}.feather')
    series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_{train_test}.feather').values
    
    y = None
    if train_test == 'train':
        y = pd.read_csv(f'{input_path}/{train_test}_labels.csv')

    dynamic_feature_names = series.drop(['customer_ID', 'S_2'], axis=1).columns.tolist()
    args.num_nodes = len(dynamic_feature_names)

    # 1. Initialize Model first to get Tokenizer
    print("Initializing Model and Tokenizer...")
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
    
    dataset = Amex_Dataset(
        series, 
        series_idx, 
        tokenizer=gen_prompt_emb.tokenizer,
        feature_names=dynamic_feature_names,
        max_len=gen_prompt_emb.max_len,
        df_y=y
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False, 
        collate_fn=dataset.collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    emb_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/emb_04/{train_test}/'
    os.makedirs(emb_path, exist_ok=True)

    bar = tqdm(dataloader)
    for data in bar:
        idxs = data['batch_idx']
        
        # Flatten inputs: (Batch, Seq, Len) -> (Batch*Seq, Len)
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
        
        # Save results
        for i, customer_id in enumerate(idxs):
            file_path = f"{emb_path}/{customer_id}.h5"
            customer_emb = embeddings_batch[i].detach().cpu().numpy()
            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('stacked_embeddings', data=customer_emb)

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