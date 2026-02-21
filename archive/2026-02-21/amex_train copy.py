import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
from torch.utils.data import DataLoader
# from data_provider.data_loader_emb import Dataset_ETT_minute
from model.TimeKD import Dual
from utils.kd_loss import KDLoss
from utils.metrics import MSE, MAE, metric
import faulthandler
from tqdm import tqdm
import pandas as pd
from utils.tools import StandardScaler
import h5py
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold


faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="Amex", help="data path")
    parser.add_argument("--sampling", type=str, default='100pct', help="Sampling ratio")
    parser.add_argument("--channel", type=int, default=512, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=223, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=13, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=1, help="out_len")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--lrate", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--feature_w", type=float, default=0.01, help="weight of feature kd loss")
    parser.add_argument("--fcst_w", type=float, default=1, help="weight of forecast loss")
    parser.add_argument("--recon_w", type=float, default=0.5, help="weight of reconstruction loss")
    parser.add_argument("--att_w", type=float, default=0.01, help="weight of attention kd loss")
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument("--epochs", type=int, default=10, help="")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument(
        "--es_patience",
        type=int,
        default=5,
        help="quit if no improvement after this many iterations",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/",
        # default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
        help="save path",
    )
    return parser.parse_args()

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
        
        emb_path = f"amex_emb/{args.sampling}/train/"
        file_path = os.path.join(emb_path, f"{idx}.h5")
        # print(f'file_path: {file_path}')

        with h5py.File(file_path, 'r') as hf:
            emb_data = hf['stacked_embeddings'][:]
            emb_tensor = torch.from_numpy(emb_data)

        if self.df_y is not None:
            label = self.df_y.loc[idx,[self.label_name]].values
            return {
                    'SERIES': series,#np.concatenate([series,series_],axis=1),
                    # 'FEATURE': np.concatenate([feature,feature_]),
                    'LABEL': label,
                    'time_ref': time_ref,
                    'idx': idx,
                    'emb_tensor': emb_tensor,
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
        Padding to same size.
        """

        batch_size = len(batch)
        batch_series = torch.zeros((batch_size, 13, batch[0]['SERIES'].shape[1]))
        batch_mask = torch.zeros((batch_size, 13))
        # batch_feature = torch.zeros((batch_size, batch[0]['FEATURE'].shape[0]))
        batch_y = torch.zeros(batch_size)
        batch_time_ref = np.array([sample['time_ref'] for sample in batch])
        batch_idx = np.array([sample['idx'] for sample in batch])
        batch_emb_tensor = None

        for i, item in enumerate(batch):
            v = item['SERIES']
            batch_series[i, :v.shape[0], :] = torch.tensor(v).float()
            batch_mask[i,:v.shape[0]] = 1.0
            # v = item['FEATURE'].astype(np.float32)
            # batch_feature[i] = torch.tensor(v).float()
            if self.df_y is not None:
                v = item['LABEL'].astype(np.float32)
                batch_y[i] = torch.tensor(v).float()
                batch_emb_tensor = torch.stack([sample['emb_tensor'] for sample in batch], dim=0) 

        return {'batch_series':batch_series
                ,'batch_mask':batch_mask
                # ,'batch_feature':batch_feature
                ,'batch_y':batch_y
                ,'batch_time_ref':batch_time_ref
                ,'batch_idx':batch_idx
                ,'batch_emb_tensor':batch_emb_tensor
                }

class trainer:
    def __init__(
        self,
        scaler,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        e_layer,
        head,
        lrate,
        wdecay,
        feature_w,
        fcst_w,
        recon_w,
        att_w,
        device,
        epochs
    ):
        self.model = Dual(
            device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len, 
            dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, head=head
        )
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 100), eta_min=1e-8, verbose=True)
        self.MSE = MSE
        self.MAE = MAE
        self.clip = 5
        self.scaler = scaler
        self.device = device

        self.feature_loss = 'smooth_l1'  
        # self.fcst_loss = 'smooth_l1'
        self.fcst_loss = 'bce'
        self.recon_loss = 'smooth_l1'
        self.att_loss = 'smooth_l1'   
        self.fcst_w = 1
        self.recon_w = 0.5
        self.feature_w = 0.1     
        self.att_w = 0.01
        self.criterion = KDLoss(self.feature_loss, self.fcst_loss, self.recon_loss, self.att_loss,  self.feature_w,  self.fcst_w,  self.recon_w,  self.att_w)

        # print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))
        # print(self.model)

    def train(self, x, y, emb):
        self.model.train()
        self.optimizer.zero_grad()
        ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(x, emb)
        
        # print(ts_enc)
        # print(prompt_enc)
        # print(ts_out)
        # print(prompt_out)
        # print(y)

        ## special handle in case of batch size = 1
        if x.shape[0] == 1:
            y =  torch.tensor([y]).to(device)
        
        loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip) 
        self.optimizer.step() 
        return loss.item(), ts_out

    def eval(self, x, y, emb):
        self.model.eval()

        ## special handle in case of batch size = 1
        if x.shape[0] == 1:
            y =  torch.tensor([y]).to(device)

        with torch.no_grad():
            ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(x, emb)
            loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
        return loss.item(), ts_out


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

args = parse_args()
input_path = f'../../000_data/amex/13month_{args.sampling}'
print(f'input_path: {input_path}')

seed_it(args.seed)
device = torch.device(args.device)

path = os.path.join(args.save, args.data_path, 
                    f"{args.pred_len}_{args.channel}_{args.e_layer}_{args.lrate}_{args.dropout_n}_{args.seed}_{args.att_w}/")
if not os.path.exists(path):
    os.makedirs(path)
    
his_loss = []
val_time = []
train_time = []
test_time = []
print(args)

engine = trainer(
    scaler=StandardScaler,
    channel=args.channel,
    num_nodes=args.num_nodes,
    seq_len=args.seq_len,
    pred_len=args.pred_len,
    dropout_n=args.dropout_n,
    d_llm=args.d_llm,
    e_layer=args.e_layer,
    head=args.head,
    lrate=args.lrate,
    wdecay=args.weight_decay,
    feature_w=args.feature_w,
    fcst_w=args.fcst_w,
    recon_w=args.recon_w,
    att_w=args.att_w,
    device=device,
    epochs=args.epochs
    )

def main_train():
    print(f'Training...')

    loss = 9999999
    epochs_since_best_mse = 0

    trainval_series     = pd.read_feather(f'{input_path}/df_nn_series_train.feather')
    trainval_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_train.feather').values
    trainval_y = pd.read_csv(f'{input_path}/train_labels.csv')

    skf = StratifiedKFold(n_splits = 2, shuffle=True, random_state=42)

    fold1, fold2 = skf.split(trainval_y,trainval_y['target'])
    trn_index, val_index = fold1[0], fold1[1]
   
    train_dataset = Amex_Dataset(trainval_series,[trainval_series_idx[i] for i in trn_index],trainval_y)
    train_dataloader = DataLoader(train_dataset,batch_size=4,shuffle=True, drop_last=False, collate_fn=train_dataset.collate_fn,num_workers=args.num_workers)

    validation_dataset = Amex_Dataset(trainval_series,[trainval_series_idx[i] for i in val_index],trainval_y)
    validation_dataloader = DataLoader(validation_dataset,batch_size=4,shuffle=False, drop_last=False, collate_fn=train_dataset.collate_fn,num_workers=args.num_workers)

    print("Start training...", flush=True)

    for i in range(1, args.epochs + 1):
        t1 = time.time()
        train_loss = []
        train_outputs = []
        train_ys = []
        
        for iter, data in enumerate(tqdm(train_dataloader)):
            y = data['batch_y']
            x = data['batch_series']
            emb_tensor = data['batch_emb_tensor']

            trainx = torch.Tensor(x).to(device).float()
            trainy = torch.Tensor(y).to(device).float()
            emb = torch.Tensor(emb_tensor).to(device).float()

            ## debug print
            # print(f'trainx shape: {trainx.shape}')
            # print(f'trainy shape: {trainy.shape}')
            # print(f'emb shape: {emb.shape}')

            curr_train_loss, train_pred_y = engine.train(trainx, trainy, emb)
            train_loss.append(curr_train_loss)

            ## special handle in case of batch size = 1
            if trainx.shape[0] == 1:
                train_pred_y =  torch.tensor([train_pred_y]).to(device)

            train_outputs.append(train_pred_y)
            train_ys.append(trainy)


        t2 = time.time()
        log = "Epoch: {:03d}, Training Time: {:.4f} secs"
        print(log.format(i, (t2 - t1)))
        train_time.append(t2 - t1)

        # Validation
        val_loss = []
        val_outputs = []
        val_ys = []
        s1 = time.time()

        for iter, data in enumerate(validation_dataloader):
            y = data['batch_y']
            x = data['batch_series']
            emb_tensor = data['batch_emb_tensor']

            valx = torch.Tensor(x).to(device).float()
            valy = torch.Tensor(y).to(device).float()
            emb = torch.Tensor(emb_tensor).to(device).float()

            curr_val_loss, val_pred_y = engine.eval(valx, valy, emb)
            val_loss.append(curr_val_loss)

            ## special handle in case of batch size = 1
            if valx.shape[0] == 1:
                val_pred_y =  torch.tensor([val_pred_y]).to(device)

            val_outputs.append(val_pred_y)
            val_ys.append(valy)

        s2 = time.time()
        log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)

       
        ## calculate train metrics
        train_pre = torch.cat(train_outputs, dim=0)
        train_y = torch.cat(train_ys, dim=0)
        train_real = torch.Tensor(train_y).to(device).float()

        # print('train_pre shape:', train_pre.shape)
        # print('train_real shape:', train_real.shape)

        pred = train_pre.squeeze().to(device)
        real = train_real.to(device)
        score = metric(pred.detach(), real.detach())
        print(f'train score: {score}')

        # print('pred shape:', pred.shape)
        # print('real shape:', real.shape)

        ## calculate val metrics
        val_pre = torch.cat(val_outputs, dim=0)
        val_y = torch.cat(val_ys, dim=0)
        val_real = torch.Tensor(val_y).to(device).float()

        # print('val_pre shape:', val_pre.shape)
        # print('test_real shape:', val_real.shape)

        pred = val_pre.squeeze().to(device)
        real = val_real.to(device)
        score = metric(pred.detach(), real.detach())
        print(f'val score: {score}')

        # print('pred shape:', pred.shape)
        # print('real shape:', real.shape)

        ## Log loss
        mtrain_loss = np.mean(train_loss)
        mvalid_loss = np.mean(val_loss)


        his_loss.append(mvalid_loss)
        print("-----------------------")

        log = "Epoch: {:03d}, Train Loss: {:.4f}"
        print(
            log.format(i, mtrain_loss),
            flush=True,
        )
        log = "Epoch: {:03d}, Valid Loss: {:.4f}"
        print(
            log.format(i, mvalid_loss),
            flush=True,
        )

        if mvalid_loss < loss:
            print("###Update tasks appear###")

            loss = mvalid_loss
            torch.save(engine.model.state_dict(), path + "best_model.pth")
            bestid = i
            epochs_since_best_mse = 0
            print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
            print("epoch: ", i)
        else:
            epochs_since_best_mse += 1
            print(f"No update. epochs_since_best_mse: {epochs_since_best_mse}")

        engine.scheduler.step()

        # if epochs_since_best_mse >= args.es_patience and i >= args.epochs//2: # early stop
        if epochs_since_best_mse >= args.es_patience and i >= min(args.epochs//2, 10): # early stop
            break

    # Output consumption
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Validation Time: {:.4f} secs".format(np.mean(val_time)))

    # Test
    print("Training ends")
    print("The epoch of the best result：", bestid)
    print("The valid loss of the best model", str(round(his_loss[bestid - 1], 4)))
   

def main_test():

    test_series     = pd.read_feather(f'{input_path}/df_nn_series_test.feather')
    test_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_test.feather').values
    test_y = pd.read_csv(f'{input_path}/test_labels.csv')['target']

    test_dataset = Amex_Dataset(test_series,test_series_idx)
    # test_dataloader = DataLoader(test_dataset,batch_size=4,shuffle=True, drop_last=True, collate_fn=test_dataset.collate_fn,num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset,batch_size=4,shuffle=False, drop_last=False, collate_fn=test_dataset.collate_fn,num_workers=args.num_workers)

    engine.model.load_state_dict(torch.load(path + "best_model.pth"), strict=False)
    
    test_outputs = []
    
    test_start_time = time.time()
    for iter, data in enumerate(test_dataloader):
        x = data['batch_series']

        testx = torch.Tensor(x).to(device).float()

        # print('testx shape:', testx.shape)

        with torch.no_grad():
            preds = engine.model(testx, None)
        
        ## special handle in case of batch size = 1
        if testx.shape[0] == 1:
            pred_y =  torch.tensor([preds[2]]).to(device)
        else:
            pred_y = preds[2]
        # print('preds len:', pred_y.shape)
        # print('preds:', pred_y)
        test_outputs.append(pred_y)


    test_pre = torch.cat(test_outputs, dim=0)
    test_real = torch.Tensor(test_y).to(device).float()

    print('test_pre shape:', test_pre.shape)
    print('test_real shape:', test_real.shape)

    pred = test_pre.squeeze().to(device)
    real = test_real.to(device)
    score = metric(pred, real)
    print(f'score: {score}')

    # print('pred shape:', pred.shape)
    # print('real shape:', real.shape)   

    test_end_time = time.time()
    print(f"Test time (total): {test_end_time - test_start_time:.4f} seconds")

    ## output test result to log file
    test_result_df = pd.DataFrame()
    test_result_df['lr'] = [args.lrate]
    test_result_df['score'] = [score]

    if not os.path.exists('./experiment_log.csv'):
        test_result_df.to_csv('./experiment_log.csv',index=False)
    else:
        test_result_df.to_csv('./experiment_log.csv',index=False,header=None,mode='a') 


if __name__ == "__main__":
    t1 = time.time()
    main_train()
    t2 = time.time()
    print("Total train time spent: {:.4f}".format(t2 - t1))

    t1 = time.time()
    main_test()
    t2 = time.time()
    print("Total test time spent: {:.4f}".format(t2 - t1))
