import torch
import os
import time
import h5py
import argparse
from torch.utils.data import DataLoader

from amex_generate_embedding import GenPromptEmb
from tqdm import tqdm

import pandas as pd
import numpy as np
from datetime import datetime

class Amex_Dataset:
    # def __init__(self,df_series,df_feature,uidxs,df_y=None):
    def __init__(self,df_series,uidxs,df_y=None,label_name = 'target',id_name = 'customer_ID'):
        self.df_series = df_series
        # self.df_feature = df_feature
        self.df_y = df_y
        self.uidxs = uidxs
        self.label_name = label_name
        self.id_name = id_name

    def __len__(self):
        return (len(self.uidxs))

    def __getitem__(self, index):
        i1,i2,idx = self.uidxs[index]
        series = self.df_series.iloc[i1:i2+1,1:].drop(['S_2'],axis=1).values
        time_ref = self.df_series.iloc[i1:i2+1,1:]['S_2']
        # series = self.df_series.iloc[i1:i2+1,1:].drop(['year_month','S_2'],axis=1).values

        if len(series.shape) == 1:
            series = series.reshape((-1,)+series.shape[-1:])
        # series_ = series.copy()
        # series_[series_!=0] = 1.0 - series_[series_!=0] + 0.001
        # feature = self.df_feature.loc[idx].values[1:]
        # feature_ = feature.copy()
        # feature_[feature_!=0] = 1.0 - feature_[feature_!=0] + 0.001
        if self.df_y is not None:
            label = self.df_y.loc[idx,[self.label_name]].values
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    # 'FEATURE': np.concatenate([feature,feature_]),
                    'LABEL': label,
                    'time_ref': time_ref,
                    'idx': idx,
                    }
        else:
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    # 'FEATURE': np.concatenate([feature,feature_]),
                    'time_ref': time_ref,
                    'idx': idx,
                    }

    def collate_fn(self, batch):
        """
        Padding to same size for tensors.
        Keeping time_ref as a list for inhomogeneous shapes.
        """

        batch_size = len(batch)
        batch_series = torch.zeros((batch_size, 13, batch[0]['SERIES'].shape[1]))
        batch_mask = torch.zeros((batch_size, 13))
        # batch_feature = torch.zeros((batch_size, batch[0]['FEATURE'].shape[0]))
        batch_y = torch.zeros((batch_size, 1))

        # Keep as a list to avoid "inhomogeneous shape" ValueError
        batch_time_ref = [sample['time_ref'].values for sample in batch]
        batch_idx = np.array([sample['idx'] for sample in batch])

        for i, item in enumerate(batch):
            v = item['SERIES']
            batch_series[i, :v.shape[0], :] = torch.tensor(v).float()
            batch_mask[i,:v.shape[0]] = 1.0
            # v = item['FEATURE'].astype(np.float32)
            # batch_feature[i] = torch.tensor(v).float()
            if self.df_y is not None:
                v = item['LABEL'].astype(np.float32)
                batch_y[i] = torch.tensor(v).float()

        return {'batch_series':batch_series
                ,'batch_mask':batch_mask
                # ,'batch_feature':batch_feature
                ,'batch_y':batch_y
                ,'batch_time_ref':batch_time_ref
                ,'batch_idx':batch_idx
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

def save_train_embeddings(args, train_test = 'train'):
    print(f'save_train_embeddings')

    input_path = None
    input_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/'
    
    series     = pd.read_feather(f'{input_path}/df_nn_series_{train_test}.feather')
    series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_{train_test}.feather').values
    
    if train_test == 'train':
        y = pd.read_csv(f'{input_path}/{train_test}_labels.csv')
        dataset = Amex_Dataset(series,series_idx,y)
        dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False, drop_last=False, collate_fn=dataset.collate_fn,num_workers=args.num_workers)
    elif train_test == 'test':
        dataset = Amex_Dataset(series,series_idx)
        dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False, drop_last=False, collate_fn=dataset.collate_fn,num_workers=args.num_workers)
    else:
        print(f'train_test: {train_test}')
        exit()

    dynamic_feature_names = series.drop(['customer_ID', 'S_2'], axis=1).columns.tolist()
    args.num_nodes = len(dynamic_feature_names)

    gen_prompt_emb = GenPromptEmb(
        # data_path=args.data_path,
        model_name=args.model_name,
        num_nodes=args.num_nodes,
        device=args.device,
        input_len=args.input_len,
        output_len=args.output_len,
        d_model=args.d_model,
        l_layer=args.l_layers,
        feature_names=dynamic_feature_names,
    ).to(args.device)

    emb_path = f'../../000_data/amex/{args.data_type}_{args.sampling}/emb_04/{train_test}/'
    os.makedirs(emb_path, exist_ok=True)

    # embeddings_list = []

    bar = tqdm(dataloader)
    for data in bar:
        y = data['batch_y'].to(args.device)
        # x = data['batch_series'].to(args.device)
        x = data['batch_series'] ## keep on cpu because subsequence string operation is on cpu
        time_ref = data['batch_time_ref']
        idxs = data['batch_idx']
        
        # hd_start_date = f'{time_ref[0,0]}'
        # hd_end_date = f'{time_ref[0,-1]}'
        # print(hd_start_date, hd_end_date)

        # print(f'x_shape: {x.shape}')
        # print(f'y_shape: {y.shape}')
        # print(f'time_ref_shape: {time_ref.shape}')
        # exit()

        # embeddings = gen_prompt_emb.generate_embeddings(x, y, time_ref)
        # print(f'embeddings_shape: {embeddings.shape}')

        # file_path = f"{emb_path}/{idx[0]}.h5"
        # with h5py.File(file_path, 'w') as hf:
        #     hf.create_dataset('stacked_embeddings', data=embeddings.detach().cpu().numpy())

        # This processes BATCH_SIZE * NUM_NODES in one GPU pass
        embeddings_batch = gen_prompt_emb.generate_embeddings(x, y, time_ref)

        # Iterate through the batch to save individual files
        for i, customer_id in enumerate(idxs):
            file_path = f"{emb_path}/{customer_id}.h5"
            # Extract the specific embedding for this customer
            # Result of generate_embeddings is (Batch, Nodes, D_model)
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
