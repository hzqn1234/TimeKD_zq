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
from model.CAI_model import Amodel
from utils.kd_loss import KDLoss
from utils.metrics import MSE, MAE, metric
import faulthandler
from tqdm import tqdm
import pandas as pd
from utils.tools import StandardScaler
import h5py
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from datetime import datetime


faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="Amex", help="data path")
    parser.add_argument("--sampling", type=str, default='100pct', help="Sampling ratio")
    parser.add_argument("--data_type", type=str, default='original')
    parser.add_argument("--channel", type=int, default=512, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=223, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=13, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=1, help="out_len")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
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
    parser.add_argument('--kfold', type=int, default=5, help='number of folds')
    parser.add_argument('--train', action='store_true', help='flag to train')
    parser.add_argument('--test', action='store_true', help='flag to test')
    parser.add_argument('--predict', action='store_true', help='flag to predict')
    parser.add_argument('--submit', action='store_true', help='flag to submit')
    parser.add_argument("--emb_version", type=str, default="v1")
    parser.add_argument("--remark", type=str, default="")
    
    
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
        self.is_train = df_y is not None

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
        
        if self.is_train:
            # emb_path = f"amex_emb/{args.data_type}/{args.sampling}/train/"
            file_path = os.path.join(emb_path, f"{idx}.h5")
            # print(f'file_path: {file_path}')

            with h5py.File(file_path, 'r') as hf:
                emb_data = hf['stacked_embeddings'][:]
                emb_tensor = torch.from_numpy(emb_data)


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
        # batch_time_ref = np.array([sample['time_ref'] for sample in batch])
        # batch_time_ref = [sample['time_ref'] for sample in batch]
        batch_idx = np.array([sample['idx'] for sample in batch])
        batch_emb_tensor = None

        for i, item in enumerate(batch):
            v = item['SERIES']
            batch_series[i, :v.shape[0], :] = torch.tensor(v).float()
            batch_mask[i,:v.shape[0]] = 1.0
            # v = item['FEATURE'].astype(np.float32)
            # batch_feature[i] = torch.tensor(v).float()

            if self.is_train:
                v = item['LABEL'].astype(np.float32)
                batch_y[i] = torch.tensor(v).float()
        
        if self.is_train:
            batch_emb_tensor = torch.stack([sample['emb_tensor'] for sample in batch], dim=0) 
                

        return {'batch_series':batch_series
                ,'batch_mask':batch_mask
                # ,'batch_feature':batch_feature
                ,'batch_y':batch_y
                # ,'batch_time_ref':batch_time_ref
                ,'batch_idx':batch_idx
                ,'batch_emb_tensor':batch_emb_tensor
                }

class Criterion:
    def __init__(self):
        self.feature_loss = 'smooth_l1'  
        # self.fcst_loss = 'smooth_l1'
        self.fcst_loss = 'bce'
        self.recon_loss = 'smooth_l1'
        self.att_loss = 'smooth_l1'   
        self.fcst_w    = args.fcst_w    ## 1
        self.recon_w   = args.recon_w   ## 0.5
        self.feature_w = args.feature_w ## 0.1     
        self.att_w     = args.att_w     ## 0.01
        self.criterion = KDLoss(self.feature_loss, self.fcst_loss, self.recon_loss, self.att_loss,  self.feature_w,  self.fcst_w,  self.recon_w,  self.att_w)

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
        device,
        epochs
    ):
        # self.model = Dual(
        #     device=device, channel=channel, num_nodes=num_nodes, seq_len=seq_len, pred_len=pred_len, 
        #     dropout_n=dropout_n, d_llm=d_llm, e_layer=e_layer, head=head
        # )

        # series_dim, feature_dim, target_num, hidden_num, hidden_dim, drop_rate=0.5, use_series_oof=False)
        self.model = Amodel(223, 16, 1, 3, 128)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 100), eta_min=1e-8, verbose=True)
        self.MSE = MSE
        self.MAE = MAE
        self.clip = 5
        self.scaler = scaler
        self.device = device
        self.criterion = criterion.criterion

        # print("The number of trainable parameters: {}".format(self.model.count_trainable_params()))
        print("The number of parameters: {}".format(self.model.param_num()))
        # print(self.model)

        self.model.to(self.device)

    def train(self, x, y, emb, m=None):
        self.model.train()
        self.optimizer.zero_grad()
        # ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(x, emb)
        ts_out = self.model(x,m)
        
        ## special handle in case of batch size = 1
        if x.shape[0] == 1:
            y =  torch.tensor([y]).to(device)
        
        # loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
        loss = self.criterion(None, None, ts_out, None, None, None, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip) 
        self.optimizer.step() 
        return loss.item(), ts_out

    def eval(self, x, y, emb,m):
        self.model.eval()

        ## special handle in case of batch size = 1
        if x.shape[0] == 1:
            y =  torch.tensor([y]).to(device)

        with torch.no_grad():
            # ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(x, emb)
            # loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
            ts_out = self.model(x,m)
            loss = self.criterion(None, None, ts_out, None, None, None, y)
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
INPUT_PATH  = f'../../000_data/amex/{args.data_type}_{args.sampling}'
emb_path    = f'../../000_data/amex/{args.data_type}_{args.sampling}/emb_01/train/'
print(f'INPUT_PATH: {INPUT_PATH}')
print(f'emb_path: {emb_path}')

seed_it(args.seed)
device = torch.device(args.device)

model_specs_template =    "{args.data_type}_{args.sampling}_{args.lrate}_{args.seed}_{args.batch_size}_{args.es_patience}" \
                       +  "_{args.channel}_{args.e_layer}_{args.dropout_n}" \
                       +  "_{args.feature_w}_{args.fcst_w}_{args.recon_w}_{args.att_w}"
model_specs          =   f"{args.data_type}_{args.sampling}_{args.lrate}_{args.seed}_{args.batch_size}_{args.es_patience}" \
                       + f"_{args.channel}_{args.e_layer}_{args.dropout_n}" \
                       + f"_{args.feature_w}_{args.fcst_w}_{args.recon_w}_{args.att_w}"
model_path = os.path.join(args.save, args.data_path, model_specs, '')

print(f'model_specs: {model_specs}')
print(f'model_path: {model_path}')
if not os.path.exists(model_path):
    os.makedirs(model_path)
    

# val_time = []
# train_time = []
# test_time = []
print(args)

criterion = Criterion()

def main_train():
    train_start_time = datetime.now()
    print(f"Start training at {train_start_time}")
    
    input_path = INPUT_PATH
    trainval_series     = pd.read_feather(f'{input_path}/df_nn_series_train.feather')
    trainval_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_train.feather').values
    trainval_y = pd.read_csv(f'{input_path}/train_labels.csv')

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for fold_index, (trn_index, val_index) in enumerate(skf.split(trainval_y,trainval_y['target'])):
        fold_train_start_time =  datetime.now()
        print(f"Start training... Fold {fold_index} at {fold_train_start_time}")

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
            device=device,
            epochs=args.epochs
        )

        train_dataset = Amex_Dataset(trainval_series,[trainval_series_idx[i] for i in trn_index],trainval_y)
        train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True, drop_last=False, collate_fn=train_dataset.collate_fn,num_workers=args.num_workers)
        validation_dataset = Amex_Dataset(trainval_series,[trainval_series_idx[i] for i in val_index],trainval_y)
        validation_dataloader = DataLoader(validation_dataset,batch_size=args.batch_size,shuffle=False, drop_last=False, collate_fn=train_dataset.collate_fn,num_workers=args.num_workers)       

        loss = 9999999
        epochs_since_best_mse = 0
        best_score=None
        his_loss = []
        early_stopped = False

        for i in range(1, args.epochs + 1):
            print(f"Staring training: fold {fold_index} - Epoch {i}")
            t1 = time.time()
            train_loss = []
            train_outputs = []
            train_ys = []
            
            for _, data in enumerate(tqdm(train_dataloader)):
                y = data['batch_y']
                x = data['batch_series']
                mask = data['batch_mask'].to(device)
                emb_tensor = data['batch_emb_tensor']

                trainx = torch.Tensor(x).to(device).float()
                trainy = torch.Tensor(y).to(device).float()
                emb = torch.Tensor(emb_tensor).to(device).float()

                curr_train_loss, train_pred_y = engine.train(trainx, trainy, emb, mask)
                train_loss.append(curr_train_loss)

                ## special handle in case of batch size = 1
                if trainx.shape[0] == 1:
                    train_pred_y =  torch.tensor([train_pred_y]).to(device)

                train_outputs.append(train_pred_y)
                train_ys.append(trainy)


            t2 = time.time()
            log = "Epoch: {:03d}, Training Time: {:.4f} secs"
            print(log.format(i, (t2 - t1)))
            # train_time.append(t2 - t1)

            # Validation
            val_loss = []
            val_outputs = []
            val_ys = []
            s1 = time.time()

            for _, data in enumerate(tqdm(validation_dataloader)):
                y = data['batch_y']
                x = data['batch_series']
                mask = data['batch_mask'].to(device)
                emb_tensor = data['batch_emb_tensor']

                valx = torch.Tensor(x).to(device).float()
                valy = torch.Tensor(y).to(device).float()
                emb = torch.Tensor(emb_tensor).to(device).float()

                curr_val_loss, val_pred_y = engine.eval(valx, valy, emb,mask)
                val_loss.append(curr_val_loss)

                ## special handle in case of batch size = 1
                if valx.shape[0] == 1:
                    val_pred_y =  torch.tensor([val_pred_y]).to(device)

                val_outputs.append(val_pred_y)
                val_ys.append(valy)

            s2 = time.time()
            log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
            print(log.format(i, (s2 - s1)))
        
            ## calculate train metrics
            train_pre = torch.cat(train_outputs, dim=0)
            train_y = torch.cat(train_ys, dim=0)
            train_real = torch.Tensor(train_y).to(device).float()

            pred = train_pre.squeeze().to(device)
            real = train_real.to(device)
            train_score = metric(pred.detach(), real.detach())
            print(f'train_score: {train_score}')

            ## calculate val metrics
            val_pre = torch.cat(val_outputs, dim=0)
            val_y = torch.cat(val_ys, dim=0)
            val_real = torch.Tensor(val_y).to(device).float()

            pred = val_pre.squeeze().to(device)
            real = val_real.to(device)
            val_score = metric(pred.detach(), real.detach())
            print(f'val_score: {val_score}')

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
                torch.save(engine.model.state_dict(), model_path + f"best_model_fold_{fold_index}.pth")
                bestid = i
                epochs_since_best_mse = 0
                best_score = val_score
                print("Updating! Valid Loss:{:.4f}".format(mvalid_loss), end=", ")
                print("epoch: ", i)
            else:
                epochs_since_best_mse += 1
                print(f"No update. epochs_since_best_mse: {epochs_since_best_mse}")

            engine.scheduler.step()

            # if epochs_since_best_mse >= args.es_patience and i >= min(args.epochs//2, 10): # early stop
            if epochs_since_best_mse >= args.es_patience: # early stop
                early_stopped = True
                print("Early Stop \n")
                break

        fold_train_end_time = datetime.now()
        fold_train_duration = fold_train_end_time - fold_train_start_time
        print(f"Training ends... Fold {fold_index} at {fold_train_end_time}")
        print(f"Total train time spent for fold {fold_index}: {fold_train_duration}")
        print(f"The epoch of the best result：{bestid}")
        print(f"The valid loss of the best model {str(round(his_loss[bestid - 1], 4))} \n", )
    
        ## output train result to log file
        log_df = create_log_df()
        log_df['fold_index'] = [fold_index]
        log_df['amex_metric'] = [f"{best_score[0]:.6g}"]
        log_df['AUC'] = [f"{best_score[1]:.6g}"]
        log_df['early_stopped'] = [early_stopped]
        log_df['fold_train_start_time'] = [fold_train_start_time.strftime('%Y-%m-%d %H:%M:%S')]
        log_df['fold_train_end_time'] = [fold_train_end_time.strftime('%Y-%m-%d %H:%M:%S')]
        log_df['fold_train_duration'] = [fold_train_duration]
        log_df = save_log(log_type='train',log_df=log_df)
    
    train_end_time = datetime.now()
    train_duration = train_end_time - train_start_time
    print(f"Training ends at {train_end_time}")
    print(f"Total train time spent for all folds: {train_duration} \n")    
    return log_df


def main_test(is_predict=False):
    test_start_time =  datetime.now()
    print(f"Start Testing at {test_start_time}")
    print(f'is_predict={is_predict}')

    if is_predict:
        input_path = f'../../000_data/amex/original_100pct'
        print(f'predict input_path: {input_path}')
    else:
        input_path = INPUT_PATH
        test_y     = pd.read_csv(f'{input_path}/test_labels.csv')['target']

    test_series     = pd.read_feather(f'{input_path}/df_nn_series_test.feather')
    test_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_test.feather').values


    test_dataset = Amex_Dataset(test_series,test_series_idx)
    test_dataloader = DataLoader(test_dataset,batch_size=args.batch_size * 2,shuffle=False, drop_last=False, collate_fn=test_dataset.collate_fn,num_workers=args.num_workers)

    models = []
    test_outputs = []

    print(f'Load models...')
    for fold_index in range(args.kfold):
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
            device=device,
            epochs=args.epochs
        )
        
        model = engine.model
        model.load_state_dict(torch.load(model_path + f"best_model_fold_{fold_index}.pth"), strict=False)
        model.eval()
        models.append(model)
              

    for _, data in enumerate(tqdm(test_dataloader)):
        x = data['batch_series']
        mask = data['batch_mask'].to(device)

        testx = torch.Tensor(x).to(device).float()

        with torch.no_grad():
            ## special handle in case of batch size = 1
            if testx.shape[0] == 1:
                pred_y = torch.tensor([torch.stack([m(testx, mask) for m in models],0).mean(0)]).to(device)
            else:
                pred_y = torch.stack([m(testx, mask) for m in models],0).mean(0)
            # if testx.shape[0] == 1:
            #     pred_y = torch.tensor([torch.stack([m(testx, mask)[2] for m in models],0).mean(0)]).to(device)
            # else:
            #     pred_y = torch.stack([m(testx, mask)[2] for m in models],0).mean(0)
        test_outputs.append(pred_y)

    print(pred_y.shape)
    test_pre = torch.cat(test_outputs, dim=0)
    pred = test_pre.squeeze().to(device)
    
    if not is_predict:
        test_real = torch.Tensor(test_y).to(device).float()
        real = test_real.to(device)
        test_score = metric(pred, real)
        print(f'test_score: {test_score}')
   
    else:
        sub = test_series[['customer_ID']].iloc[test_series_idx[:, 0]].copy()
        sub['prediction'] = pred.cpu().detach().numpy()
        sub.to_csv(model_path+'submission.csv.zip',index=False, compression='zip')
  
    test_end_time = datetime.now()
    test_duration = test_end_time - test_start_time
    print(f"Testing ends at {test_end_time}")
    print(f"Total test time spent: {test_duration}") 

    ## output test result to log file
    log_df = create_log_df()
    log_df['is_predict'] = [is_predict]
    if not is_predict:
        log_df['amex_metric']   = [f"{test_score[0]:.6g}"]
        log_df['AUC']           = [f"{test_score[1]:.6g}"]
        log_type='test'
    else:
        log_df['amex_metric']   = [None]
        log_df['AUC']           = [None]
        log_type='predict'
    log_df['test_start_time'] = [test_start_time.strftime('%Y-%m-%d %H:%M:%S')]
    log_df['test_end_time'] = [test_end_time.strftime('%Y-%m-%d %H:%M:%S')]
    log_df['test_duration'] = [test_duration]
    log_df = save_log(log_type=log_type,log_df=log_df)
    return log_df

def create_log_df():
    log_df = pd.DataFrame()
    log_df['model_specs'] = [model_specs]
    log_df['log_time'] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] 
    log_df['feature_w'] = [args.feature_w]
    log_df['fcst_w'] = [args.fcst_w]
    log_df['recon_w'] = [args.recon_w]
    log_df['att_w'] = [args.att_w]
    log_df['lr'] = [args.lrate]
    log_df['sampling'] = [args.sampling]
    log_df['data_type'] = [args.data_type]
    log_df['seed'] = [args.seed]
    log_df['batch_size'] = [args.batch_size]  
    log_df['es_patience'] = [args.es_patience]  
    log_df['emb_version'] = [args.emb_version]
    log_df['remark'] = [args.remark]
    return log_df

def save_log(log_type='train',log_df=None):
    if not os.path.exists(f'./experiment_log_{log_type}.csv'):
        log_df.to_csv(f'./experiment_log_{log_type}.csv',index=False)
    else:
        log_df.to_csv(f'./experiment_log_{log_type}.csv',index=False,header=None,mode='a') 
    return log_df

if __name__ == "__main__":
    log_df = None

    if args.train:
        main_train()      

    if args.test:
        if args.data_type == 'original' and args.sampling == '100pct':
            print('Skip Test for orginal_100pct')
            pass
        else:
            main_test()

    if args.predict:
        log_df = main_test(is_predict=True)

    if args.submit:
        if log_df is not None:
            submit_message = log_df.to_json(orient='records')
        else:
            submit_message = f'{model_specs_template}: {model_specs}'
        os.system(f"""kaggle competitions submit -c amex-default-prediction -f {model_path}/submission.csv.zip -m '{submit_message}'""")
        print("\n")
        pass

