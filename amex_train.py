import torch
from torch import optim
import numpy as np
import argparse
import time
import os
import random
import glob

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from torch.utils.data import DataLoader
from model.CAI_model import Amodel
from utils.kd_loss import KDLoss
from utils.metrics import MSE, MAE, metric
import faulthandler
from tqdm import tqdm
import pandas as pd
from utils.tools import StandardScaler
import h5py
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from datetime import datetime

faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"

def parse_args():
    parser = argparse.ArgumentParser()
    # ================= 新增阶段控制参数 =================
    parser.add_argument("--stage", type=int, default=0, help="0: 联合训练, 1: 仅训练Teacher(预训练), 2: 仅训练Student(蒸馏)")
    parser.add_argument("--teacher_dir", type=str, default="", help="Stage 2需要提供Stage 1训练输出的best models所在目录")
    # ====================================================
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
    parser.add_argument("--d_llm", type=int, default=896, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--feature_w", type=float, default=0.01, help="weight of feature kd loss")
    parser.add_argument("--fcst_w", type=float, default=1, help="weight of forecast loss")
    parser.add_argument("--recon_w", type=float, default=0.5, help="weight of reconstruction loss")
    parser.add_argument("--att_w", type=float, default=0.01, help="weight of attention kd loss")
    parser.add_argument("--distill_w", type=float, default=1.0, help="weight of distillation loss")
    parser.add_argument("--temperature", type=float, default=5.0, help="Temperature for Soft Label KD")
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
        help="save path",
    )
    return parser.parse_args()

class Amex_Dataset:
    def __init__(self, df_series, uidxs, df_y=None, label_name='target', id_name='customer_ID'):
        self.df_series = df_series
        self.df_y = df_y
        self.uidxs = uidxs
        self.label_name = label_name
        self.id_name = id_name
        self.is_train = df_y is not None
        
        # [V6 优化] 支持多个 Chunk 的查找表
        if self.is_train:
            chunk_files = glob.glob(os.path.join(emb_path, "train_embeddings_chunk_*.h5"))
            if not chunk_files: # Fallback backwards compatibility
                chunk_files = [os.path.join(emb_path, "train_embeddings_all.h5")]
                
            self.id_to_file_and_row = {}
            for fpath in chunk_files:
                print(f"Loading embedding index from chunk: {fpath}")
                with h5py.File(fpath, 'r') as hf:
                    all_ids = hf['customer_ids'][:]
                    
                for i, cid in enumerate(all_ids):
                    if isinstance(cid, bytes):
                        cid = cid.decode('utf-8')
                    self.id_to_file_and_row[str(cid)] = (fpath, i)
            
            print(f"Index built. Total mapped IDs across chunks: {len(self.id_to_file_and_row)}")

    def __len__(self):
        return (len(self.uidxs))

    def __getitem__(self, index):
        i1, i2, idx = self.uidxs[index]
        series = self.df_series.iloc[i1:i2+1, 1:].drop(['S_2'], axis=1).values
        time_ref = self.df_series.iloc[i1:i2+1, 1:]['S_2']

        if len(series.shape) == 1:
            series = series.reshape((-1,) + series.shape[-1:])
        
        if self.is_train:
            # 1. 查找 idx 对应的文件和行号
            mapping = self.id_to_file_and_row.get(str(idx))
            
            if mapping is None:
                raise ValueError(f"Customer ID {idx} not found in any embedding chunk!")
            
            fpath, row_idx = mapping

            # 2. Lazy load and cache HDF5 file handlers per worker process
            if not hasattr(self, 'h5_handlers'):
                self.h5_handlers = {}
            if fpath not in self.h5_handlers:
                self.h5_handlers[fpath] = h5py.File(fpath, 'r')

            # 3. Read from the cached handler
            emb_data = self.h5_handlers[fpath]['embeddings'][row_idx]
            emb_tensor = torch.from_numpy(emb_data)

            label = self.df_y.loc[idx, [self.label_name]].values
            return {
                    'SERIES': series,
                    'LABEL': label,
                    'time_ref': time_ref,
                    'idx': idx,
                    'emb_tensor': emb_tensor,
                    }
        else:
            return {
                    'SERIES': series,
                    'time_ref': time_ref,
                    'idx': idx,
                    }

    def collate_fn(self, batch):
        batch_size = len(batch)
        batch_series = torch.zeros((batch_size, 13, batch[0]['SERIES'].shape[1]))
        batch_mask = torch.zeros((batch_size, 13))
        batch_y = torch.zeros(batch_size)
        batch_idx = np.array([sample['idx'] for sample in batch])
        batch_emb_tensor = None

        for i, item in enumerate(batch):
            v = item['SERIES']
            batch_series[i, :v.shape[0], :] = torch.tensor(v).float()
            batch_mask[i, :v.shape[0]] = 1.0

            if self.is_train:
                v = item['LABEL'].astype(np.float32)
                batch_y[i] = torch.tensor(v).float()
        
        if self.is_train:
            batch_emb_tensor = torch.stack([sample['emb_tensor'] for sample in batch], dim=0).float()

        return {'batch_series': batch_series,
                'batch_mask': batch_mask,
                'batch_y': batch_y,
                'batch_idx': batch_idx,
                'batch_emb_tensor': batch_emb_tensor
                }

class Criterion:
    def __init__(self, stage):
        self.feature_loss = 'smooth_l1'  
        self.fcst_loss = 'bce'
        self.recon_loss = 'bce'
        self.att_loss = 'smooth_l1'   
        self.distill_loss = 'bce'
        
        # 动态控制各阶段的Loss权重组合
        if stage == 1:
            self.fcst_w    = 0.0
            self.recon_w   = 1.0  # Stage 1: 只优化Teacher的标签预测
            self.feature_w = 0.0
            self.att_w     = 0.0
            self.distill_w = 0.0
        elif stage == 2:
            self.fcst_w    = args.fcst_w
            self.recon_w   = 0.0  # Stage 2: 冻结Teacher，无需计算其重建Loss
            self.feature_w = args.feature_w
            self.att_w     = args.att_w
            self.distill_w = args.distill_w
        else:
            self.fcst_w    = args.fcst_w
            self.recon_w   = args.recon_w
            self.feature_w = args.feature_w
            self.att_w     = args.att_w
            self.distill_w = args.distill_w

        self.criterion = KDLoss(self.feature_loss, self.fcst_loss, self.recon_loss, self.att_loss, self.distill_loss,
                                self.feature_w,  self.fcst_w,  self.recon_w,  self.att_w, self.distill_w,
                                temperature=args.temperature
                                )

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
        epochs,
        stage=0,
        teacher_path=None
    ):
        self.MSE = MSE
        self.MAE = MAE
        self.clip = 5
        self.scaler = scaler
        self.device = device
        self.stage = stage
        self.criterion = criterion.criterion

        self.model = Amodel(223, 16, 1, 3, 128, d_llm=d_llm, device=self.device)
        self.model.to(self.device)

        # 核心修改点：根据Stage冻结不同部分的梯度
        if stage == 1:
            print("\n=== Stage 1: Freezing Student, Training Teacher ===")
            for param in self.model.input_series_block_n1.parameters(): param.requires_grad = False
            for param in self.model.transformer_encoder.parameters(): param.requires_grad = False
            for param in self.model.output_block.parameters(): param.requires_grad = False
        elif stage == 2:
            print("\n=== Stage 2: Freezing Teacher, Training Student ===")
            if teacher_path and os.path.exists(teacher_path):
                self.model.load_state_dict(torch.load(teacher_path, map_location=self.device), strict=False)
                print(f"Loaded Teacher pre-trained weights from: {teacher_path}")
            elif teacher_path is None:
                # 说明这是在 Test/Predict 阶段，正常跳过，不报 Warning
                print("Test/Predict phase initialized (Teacher weights will be loaded with the whole model).")
            else:
                print(f"WARNING: Valid Teacher path not provided for Stage 2! ({teacher_path})")
                
            for param in self.model.input_series_block_n1_t.parameters(): param.requires_grad = False
            for param in self.model.transformer_encoder_t.parameters(): param.requires_grad = False
            for param in self.model.output_block_t.parameters(): param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("The number of trainable parameters: {}".format(trainable_params))

        # 仅将需要更新的参数传递给优化器
        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(optim_params, lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=min(epochs, 100), eta_min=1e-8)

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        
        # ====================================================
        # [方案B 核心修改]: 物理破坏特权特征的"标签捷径"
        # ====================================================
        if self.stage in [1, 2] and data.get('batch_emb_tensor') is not None:
            emb = data['batch_emb_tensor'].to(self.device)
            # 1. 注入强高斯噪声 (破坏精确的数值映射)
            noise = torch.randn_like(emb) * 0.5  # 0.5 为噪声强度，可根据需要调到 1.0
            emb = emb + noise
            # 2. 强行丢弃一部分特征 (强迫模型寻找其他非作弊特征)
            emb = torch.nn.functional.dropout(emb, p=0.4, training=True) # 丢弃 40%
            data['batch_emb_tensor'] = emb
        # ====================================================

        ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(data)
        y = data['batch_y'].to(self.device)
        
        if data['batch_series'].shape[0] == 1:
            y =  torch.tensor([y]).to(self.device)
        
        loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip) 
        self.optimizer.step() 
        
        # Stage 1 时评估Teacher的预测能力以决定早停
        out_metric = prompt_out if self.stage == 1 else ts_out
        return loss.item(), out_metric

    def eval(self, data):
        self.model.eval()

        with torch.no_grad():
            ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(data)
            y = data['batch_y'].to(self.device)

            if data['batch_series'].shape[0] == 1:
                y =  torch.tensor([y]).to(self.device)

            loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
            out_metric = prompt_out if self.stage == 1 else ts_out
        return loss.item(), out_metric


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

if args.emb_version == 'v4':
    emb_path    = f'../../000_data/amex/{args.data_type}_{args.sampling}/emb_04/'
elif args.emb_version == 'v5':
    emb_path    = f'../../000_data/amex/{args.data_type}_{args.sampling}/emb_05/'
elif args.emb_version == 'v6':
    emb_path    = f'../../000_data/amex/{args.data_type}_{args.sampling}/emb_06/'
else:
    emb_path = None

print(f'INPUT_PATH: {INPUT_PATH}')
print(f'emb_path: {emb_path}')

seed_it(args.seed)
device = torch.device(args.device)

# 增加 S{args.stage}_ 前缀，确保阶段1和阶段2的模型/日志保存在不同目录
model_specs_template =    "S{args.stage}_{args.data_type}_{args.sampling}_{args.lrate}_{args.seed}_{args.batch_size}_{args.es_patience}" \
                       +  "_{args.channel}_{args.e_layer}_{args.dropout_n}" \
                       +  "_{args.feature_w}_{args.fcst_w}_{args.recon_w}_{args.att_w}_{args.distill_w}"
model_specs          =   f"S{args.stage}_{args.data_type}_{args.sampling}_{args.lrate}_{args.seed}_{args.batch_size}_{args.es_patience}" \
                       + f"_{args.channel}_{args.e_layer}_{args.dropout_n}" \
                       + f"_{args.feature_w}_{args.fcst_w}_{args.recon_w}_{args.att_w}_{args.distill_w}"
model_path = os.path.join(args.save, args.data_path, model_specs, '')

print(f'model_specs: {model_specs}')
print(f'model_path: {model_path}')
if not os.path.exists(model_path):
    os.makedirs(model_path)

print(args)

# 全局初始化 Criterion
criterion = Criterion(args.stage)

def main_train():
    train_start_time = datetime.now()
    print(f"Start training at {train_start_time}")
    
    input_path = INPUT_PATH
    trainval_series     = pd.read_feather(f'{input_path}/df_nn_series_train.feather')
    trainval_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_train.feather').values
    trainval_y = pd.read_csv(f'{input_path}/train_labels.csv')

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for fold_index, (trn_index, val_index) in enumerate(skf.split(trainval_y, trainval_y['target'])):
        fold_train_start_time =  datetime.now()
        print(f"Start training... Fold {fold_index} at {fold_train_start_time}")

        teacher_model_path = None
        if args.stage == 2:
            if args.teacher_dir:
                teacher_model_path = os.path.join(args.teacher_dir, f"best_model_fold_{fold_index}.pth")
            else:
                raise ValueError("Stage 2 必须提供 --teacher_dir 参数！")

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
            epochs=args.epochs,
            stage=args.stage,
            teacher_path=teacher_model_path
        )

        train_dataset = Amex_Dataset(trainval_series, [trainval_series_idx[i] for i in trn_index], trainval_y)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers)
        validation_dataset = Amex_Dataset(trainval_series, [trainval_series_idx[i] for i in val_index], trainval_y)
        validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=train_dataset.collate_fn, num_workers=args.num_workers)       

        loss = 9999999
        epochs_since_best_mse = 0
        best_score = None
        his_loss = []
        early_stopped = False

        for i in range(1, args.epochs + 1):
            print(f"Staring training: fold {fold_index} - Epoch {i}")
            t1 = time.time()
            train_loss = []
            train_outputs = []
            train_ys = []
            
            for _, data in enumerate(tqdm(train_dataloader)):
                curr_train_loss, train_pred_y = engine.train(data)
                train_loss.append(curr_train_loss)

                if data['batch_series'].shape[0] == 1:
                    train_pred_y =  torch.tensor([train_pred_y]).to(device)

                train_outputs.append(train_pred_y)
                train_ys.append(data['batch_y'])

            t2 = time.time()
            log = "Epoch: {:03d}, Training Time: {:.4f} secs"
            print(log.format(i, (t2 - t1)))

            # Validation
            val_loss = []
            val_outputs = []
            val_ys = []
            s1 = time.time()

            for _, data in enumerate(tqdm(validation_dataloader)):
                curr_val_loss, val_pred_y = engine.eval(data)
                val_loss.append(curr_val_loss)

                if data['batch_series'].shape[0] == 1:
                    val_pred_y =  torch.tensor([val_pred_y]).to(device)

                val_outputs.append(val_pred_y)
                val_ys.append(data['batch_y'])

            s2 = time.time()
            log = "Epoch: {:03d}, Validation Time: {:.4f} secs"
            print(log.format(i, (s2 - s1)))
        
            # calculate train metrics
            train_pre = torch.cat(train_outputs, dim=0)
            train_y = torch.cat(train_ys, dim=0)
            train_real = torch.Tensor(train_y).to(device).float()

            pred = train_pre.squeeze().to(device)
            real = train_real.to(device)
            train_score = metric(pred.detach(), real.detach())
            print(f'train_score: {train_score}')

            # calculate val metrics
            val_pre = torch.cat(val_outputs, dim=0)
            val_y = torch.cat(val_ys, dim=0)
            val_real = torch.Tensor(val_y).to(device).float()

            pred = val_pre.squeeze().to(device)
            real = val_real.to(device)
            val_score = metric(pred.detach(), real.detach())
            print(f'val_score: {val_score}')

            # Log loss
            mtrain_loss = np.mean(train_loss)
            mvalid_loss = np.mean(val_loss)

            his_loss.append(mvalid_loss)
            print("-----------------------")

            log = "Epoch: {:03d}, Train Loss: {:.4f}"
            print(log.format(i, mtrain_loss), flush=True)
            log = "Epoch: {:03d}, Valid Loss: {:.4f}"
            print(log.format(i, mvalid_loss), flush=True)

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
    
        # output train result to log file
        log_df = create_log_df()
        log_df['fold_index'] = [fold_index]
        log_df['amex_metric'] = [f"{best_score[0]:.6g}"]
        log_df['AUC'] = [f"{best_score[1]:.6g}"]
        log_df['early_stopped'] = [early_stopped]
        log_df['fold_train_start_time'] = [fold_train_start_time.strftime('%Y-%m-%d %H:%M:%S')]
        log_df['fold_train_end_time'] = [fold_train_end_time.strftime('%Y-%m-%d %H:%M:%S')]
        log_df['fold_train_duration'] = [fold_train_duration]
        log_df = save_log(log_type='train', log_df=log_df)
    
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

    test_dataset = Amex_Dataset(test_series, test_series_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size * 8, shuffle=False, drop_last=False, collate_fn=test_dataset.collate_fn, num_workers=args.num_workers)

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
            epochs=args.epochs,
            stage=args.stage,
            teacher_path=None  # Test 阶段直接用下一行载入整个模型
        )
        
        model = engine.model
        model.load_state_dict(torch.load(model_path + f"best_model_fold_{fold_index}.pth"), strict=False)
        model.eval()
        models.append(model)
              

    for _, data in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            if data['batch_series'].shape[0] == 1:
                pred_y = torch.tensor([torch.stack([m(data)[2] for m in models], 0).mean(0)]).to(device)
            else:
                pred_y = torch.stack([m(data)[2] for m in models], 0).mean(0)
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
        sub.to_csv(model_path+'submission.csv.zip', index=False, compression='zip')
  
    test_end_time = datetime.now()
    test_duration = test_end_time - test_start_time
    print(f"Testing ends at {test_end_time}")
    print(f"Total test time spent: {test_duration}") 

    # output test result to log file
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
    log_df = save_log(log_type=log_type, log_df=log_df)
    return log_df

def create_log_df():
    log_df = pd.DataFrame()
    log_df['stage'] = [args.stage]
    log_df['model_specs'] = [model_specs]
    log_df['log_time'] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] 
    log_df['feature_w'] = [args.feature_w]
    log_df['fcst_w'] = [args.fcst_w]
    log_df['recon_w'] = [args.recon_w]
    log_df['att_w'] = [args.att_w]
    log_df['distill_w'] = [args.distill_w]
    log_df['lr'] = [args.lrate]
    log_df['sampling'] = [args.sampling]
    log_df['data_type'] = [args.data_type]
    log_df['seed'] = [args.seed]
    log_df['batch_size'] = [args.batch_size]  
    log_df['es_patience'] = [args.es_patience]  
    log_df['emb_version'] = [args.emb_version]
    log_df['remark'] = [args.remark]
    return log_df

def save_log(log_type='train', log_df=None):
    log_dir = './logs/experiment_log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    file_path = f'{log_dir}/experiment_log_{log_type}.csv'    
    if not os.path.exists(file_path):
        log_df.to_csv(file_path, index=False)
    else:
        log_df.to_csv(file_path, index=False, header=None, mode='a') 
    return log_df


if __name__ == "__main__":
    log_df = None

    if args.train:
        main_train()      

    # Teacher网络不能在不带特征的Test集上测，所以限制仅在 Stage 0 或 2 测试
    if args.test:
        if args.stage == 1:
            print('>>> 跳过测试: Stage 1 为Teacher预训练，无法在没有 Embedding 的 Test 集上进行推理。')
        elif args.data_type == 'original' and args.sampling == '100pct':
            print('Skip Test for orginal_100pct')
            pass
        else:
            main_test()

    if args.predict:
        if args.stage == 1:
            print('>>> 跳过预测: Stage 1 为Teacher预训练，无法在没有 Embedding 的 Test 集上进行预测。')
        else:
            log_df = main_test(is_predict=True)

    if args.submit:
        if args.stage == 1:
            print('>>> 跳过提交: 无法提交 Stage 1 的Teacher模型预测结果。')
        else:
            if log_df is not None:
                submit_message = log_df.to_json(orient='records')
            else:
                submit_message = f'{model_specs_template}: {model_specs}'
            os.system(f"""kaggle competitions submit -c amex-default-prediction -f {model_path}/submission.csv.zip -m '{submit_message}'""")
            print("\n")
            pass