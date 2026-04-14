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
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        choices=[1, 2, 3],
        help="1: 仅训练Teacher(预训练), 2: 仅训练Student(蒸馏), 3: 仅训练Student(Baseline)"
    )
    parser.add_argument("--teacher_dir", type=str, default="", help="Stage 2需要提供Stage 1训练输出的best models所在目录")
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="Amex", help="data path")
    parser.add_argument("--sampling", type=str, default='100pct', help="Sampling ratio")
    parser.add_argument("--data_type", type=str, default='original')
    parser.add_argument("--channel", type=int, default=512, help="number of features")
    parser.add_argument("--num_nodes", type=int, default=223, help="number of nodes")
    parser.add_argument("--seq_len", type=int, default=13, help="seq_len")
    parser.add_argument("--pred_len", type=int, default=1, help="out_len")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--lrate", type=str, default=1e-5, help="learning rate")
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
    def __init__(self, df_series, uidxs, df_y=None, label_name='target', id_name='customer_ID', use_embedding=True, emb_path=None):
        self.df_series = df_series
        self.df_y = df_y
        self.uidxs = uidxs
        self.label_name = label_name
        self.id_name = id_name
        self.is_train = df_y is not None
        
        # 允许在非训练阶段使用 Embedding
        self.use_embedding = bool(use_embedding)
        self.id_to_file_and_row = {}

        if self.use_embedding:
            # 增加路径校验，必须显式传入 emb_path
            if emb_path is None:
                raise ValueError("When use_embedding is True, a valid emb_path must be provided!")
                
            # 根据是否包含 label 判断读取 train 还是 test 的 embedding
            prefix = "train" if self.is_train else "test"
            chunk_files = glob.glob(os.path.join(emb_path, f"{prefix}_embeddings_chunk_*.h5"))
            
            if not chunk_files:
                fallback_file = os.path.join(emb_path, f"{prefix}_embeddings_all.h5")
                if os.path.exists(fallback_file):
                    chunk_files = [fallback_file]

            if not chunk_files:
                raise FileNotFoundError(f"No {prefix} embedding h5 files found under {emb_path}")

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
        return len(self.uidxs)

    def __getitem__(self, index):
        i1, i2, idx = self.uidxs[index]
        series = self.df_series.iloc[i1:i2 + 1, 1:].drop(['S_2'], axis=1).values
        time_ref = self.df_series.iloc[i1:i2 + 1, 1:]['S_2']

        if len(series.shape) == 1:
            series = series.reshape((-1,) + series.shape[-1:])

        # 无论是否是 is_train，只要 use_embedding 为 True 就提取特征
        emb_tensor = None
        if self.use_embedding:
            mapping = self.id_to_file_and_row.get(str(idx))
            if mapping is None:
                raise ValueError(f"Customer ID {idx} not found in any embedding chunk!")

            fpath, row_idx = mapping

            if not hasattr(self, 'h5_handlers'):
                self.h5_handlers = {}
            if fpath not in self.h5_handlers:
                self.h5_handlers[fpath] = h5py.File(fpath, 'r')

            emb_data = self.h5_handlers[fpath]['embeddings'][row_idx]
            emb_tensor = torch.from_numpy(emb_data)

        if self.is_train:
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
                'emb_tensor': emb_tensor,
            }

    def collate_fn(self, batch):
        batch_size = len(batch)

        # Dynamically get the max sequence length in this batch instead of hardcoding 13
        max_batch_len = max([item['SERIES'].shape[0] for item in batch])

        batch_series = torch.zeros((batch_size, max_batch_len, batch[0]['SERIES'].shape[1]))
        batch_mask = torch.zeros((batch_size, max_batch_len))
        batch_y = torch.zeros(batch_size)
        batch_idx = np.array([sample['idx'] for sample in batch])

        for i, item in enumerate(batch):
            v = item['SERIES']
            batch_series[i, :v.shape[0], :] = torch.tensor(v).float()
            batch_mask[i, :v.shape[0]] = 1.0

            if self.is_train:
                label_value = item['LABEL'].astype(np.float32)
                batch_y[i] = torch.tensor(label_value).float()

        if self.use_embedding:
            batch_emb_tensor = torch.stack([sample['emb_tensor'] for sample in batch], dim=0).float()
        else:
            batch_emb_tensor = None

        return {
            'batch_series': batch_series,
            'batch_mask': batch_mask,
            'batch_y': batch_y,
            'batch_idx': batch_idx,
            'batch_emb_tensor': batch_emb_tensor,
        }


class Criterion:
    def __init__(self, stage):
        self.feature_loss = 'smooth_l1'
        self.fcst_loss = 'bce_logits'
        self.recon_loss = 'bce_logits'
        self.att_loss = 'smooth_l1'
        self.distill_loss = 'bce'

        if stage == 1:
            self.fcst_w = 0.0
            self.recon_w = 1.0
            self.feature_w = 0.0
            self.att_w = 0.0
            self.distill_w = 0.0
        elif stage == 2:
            self.fcst_w = args.fcst_w
            self.recon_w = 0.0
            self.feature_w = args.feature_w
            self.att_w = args.att_w
            self.distill_w = args.distill_w
        elif stage == 3:
            self.fcst_w = 1.0
            self.recon_w = 0.0
            self.feature_w = 0.0
            self.att_w = 0.0
            self.distill_w = 0.0
        else:
            self.fcst_w = args.fcst_w
            self.recon_w = args.recon_w
            self.feature_w = args.feature_w
            self.att_w = args.att_w
            self.distill_w = args.distill_w

        self.criterion = KDLoss(
            self.feature_loss,
            self.fcst_loss,
            self.recon_loss,
            self.att_loss,
            self.distill_loss,
            self.feature_w,
            self.fcst_w,
            self.recon_w,
            self.att_w,
            self.distill_w,
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
        teacher_path=None,
        is_training=True
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

        if stage == 1:
            if is_training:
                print("\n=== Stage 1: Freezing Student, Training Teacher ===")
            for param in self.model.input_series_block_n1.parameters():
                param.requires_grad = False
            for param in self.model.transformer_encoder.parameters():
                param.requires_grad = False
            for param in self.model.output_block.parameters():
                param.requires_grad = False

        elif stage == 2:
            if is_training:
                print("\n=== Stage 2: Freezing Teacher, Training Student ===")

            if teacher_path and os.path.exists(teacher_path):
                self.model.load_state_dict(torch.load(teacher_path, map_location=self.device), strict=False)
                if is_training:
                    print(f"Loaded Teacher pre-trained weights from: {teacher_path}")
            elif teacher_path is None:
                if is_training:
                    print("Test/Predict phase initialized (Teacher weights will be loaded with the whole model).")
            else:
                if is_training:
                    raise ValueError(f"WARNING: Valid Teacher path not provided for Stage 2! ({teacher_path})")

            for param in self.model.input_series_block_n1_t.parameters():
                param.requires_grad = False
            for param in self.model.input_series_block_n1_t_raw.parameters():
                param.requires_grad = False
            for param in self.model.transformer_encoder_t.parameters():
                param.requires_grad = False
            for param in self.model.output_block_t.parameters():
                param.requires_grad = False

        elif stage == 3:
            if is_training:
                print("\n=== Stage 3: Student-only baseline ===")

            for param in self.model.input_series_block_n1_t.parameters():
                param.requires_grad = False
            for param in self.model.input_series_block_n1_t_raw.parameters():
                param.requires_grad = False
            for param in self.model.transformer_encoder_t.parameters():
                param.requires_grad = False
            for param in self.model.output_block_t.parameters():
                param.requires_grad = False

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if is_training:
            print("The number of trainable parameters: {}".format(trainable_params))

        optim_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(optim_params, lr=lrate, weight_decay=wdecay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=min(epochs, 100), eta_min=1e-8
        )

    def train(self, data):
        self.model.train()
        self.optimizer.zero_grad()

        if self.stage in [2, 3]:
            self.model.input_series_block_n1_t.eval()
            self.model.input_series_block_n1_t_raw.eval()
            self.model.transformer_encoder_t.eval()
            self.model.output_block_t.eval()

        ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(data, stage=self.stage)
        y = data['batch_y'].to(self.device)

        if data['batch_series'].shape[0] == 1:
            y = torch.tensor([y]).to(self.device)

        loss = self.criterion(ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att, y)
        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()

        out_metric = prompt_out if self.stage == 1 else ts_out
        return loss.item(), out_metric

    def eval(self, data):
        self.model.eval()

        with torch.no_grad():
            ts_enc, prompt_enc, ts_out, prompt_out, ts_att, prompt_att = self.model(data, stage=self.stage)
            y = data['batch_y'].to(self.device)

            if data['batch_series'].shape[0] == 1:
                y = torch.tensor([y]).to(self.device)

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

def create_short_seq_train_indices(uidxs, max_len=5):
    """Augments training data by slicing long sequences to their last N months."""
    new_uidxs = []
    for i1, i2, cust_id in uidxs:
        seq_len = i2 - i1 + 1
        if seq_len <= max_len:
            # Keep natural short sequences exactly as they are
            new_uidxs.append([i1, i2, cust_id])
        else:
            # For long sequences, extract only the LAST `max_len` months
            new_i1 = i2 - max_len + 1
            new_uidxs.append([new_i1, i2, cust_id])
    return np.array(new_uidxs)

def filter_short_seq_test_indices(uidxs, max_len=5):
    """Filters test data to ONLY include natural short sequences and returns a mask."""
    new_uidxs = []
    keep_mask = []
    for i1, i2, cust_id in uidxs:
        seq_len = i2 - i1 + 1
        if seq_len <= max_len:
            new_uidxs.append([i1, i2, cust_id])
            keep_mask.append(True)
        else:
            keep_mask.append(False)
    return np.array(new_uidxs), np.array(keep_mask)


args = parse_args()
INPUT_PATH = f'../../000_data/amex/{args.data_type}_{args.sampling}'

if args.emb_version and args.emb_version.startswith('v') and args.emb_version[1:].isdigit():
    v_num = args.emb_version[1:].zfill(2)
    emb_path = f'{INPUT_PATH}/emb_{v_num}/'
else:
    emb_path = None

print(f'INPUT_PATH: {INPUT_PATH}')
print(f'emb_path: {emb_path}')

seed_it(args.seed)
device = torch.device(args.device)

stage_name_map = {
    1: "teacher",
    2: "distill",
    3: "student",
}
stage_name = stage_name_map.get(args.stage, f"stage{args.stage}")

model_specs_template =  "S{args.stage}_{stage_name}_{args.data_type}_{args.emb_version}_{args.seq_len}_{args.sampling}_{args.lrate}_{args.seed}"
model_specs          = f"S{args.stage}_{stage_name}_{args.data_type}_{args.emb_version}_{args.seq_len}_{args.sampling}_{args.lrate}_{args.seed}"
model_path = os.path.join(args.save, args.data_path, model_specs, '')

print(f'model_specs: {model_specs}')
print(f'model_path: {model_path}')
if not os.path.exists(model_path):
    os.makedirs(model_path)

print(args)

criterion = Criterion(args.stage)


def main_train():
    train_start_time = datetime.now()
    print(f"Start training at {train_start_time}")

    input_path = INPUT_PATH
    trainval_series = pd.read_feather(f'{input_path}/df_nn_series_train.feather')
    trainval_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_train.feather').values
    trainval_y = pd.read_csv(f'{input_path}/train_labels.csv')

    # Process indices for the <= 5 month model
    print(f"Original training samples: {len(trainval_series_idx)}")
    trainval_series_idx = create_short_seq_train_indices(trainval_series_idx, max_len=5)
    print(f"Augmented short-seq samples: {len(trainval_series_idx)}")

    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    fold_best_scores = []

    for fold_index, (trn_index, val_index) in enumerate(skf.split(trainval_y, trainval_y['target'])):
        fold_train_start_time = datetime.now()
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
            lrate=float(args.lrate),
            wdecay=args.weight_decay,
            device=device,
            epochs=args.epochs,
            stage=args.stage,
            teacher_path=teacher_model_path,
            is_training=True
        )

        use_embedding = args.stage in [1, 2]

        train_dataset = Amex_Dataset(
            trainval_series,
            [trainval_series_idx[i] for i in trn_index],
            trainval_y,
            use_embedding=use_embedding,
            emb_path=emb_path  # 显式传入全局解析好的 emb_path
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=train_dataset.collate_fn,
            num_workers=args.num_workers
        )

        validation_dataset = Amex_Dataset(
            trainval_series,
            [trainval_series_idx[i] for i in val_index],
            trainval_y,
            use_embedding=use_embedding,
            emb_path=emb_path  # 显式传入全局解析好的 emb_path
        )
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=validation_dataset.collate_fn,
            num_workers=args.num_workers
        )

        best_valid_loss = float("inf")
        epochs_since_best_mse = 0
        best_score = None
        his_loss = []
        early_stopped = False
        bestid = 0

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
                    train_pred_y = torch.tensor([train_pred_y]).to(device)

                train_outputs.append(train_pred_y)
                train_ys.append(data['batch_y'])

            t2 = time.time()
            print("Epoch: {:03d}, Training Time: {:.4f} secs".format(i, (t2 - t1)))

            val_loss = []
            val_outputs = []
            val_ys = []
            s1 = time.time()

            for _, data in enumerate(tqdm(validation_dataloader)):
                curr_val_loss, val_pred_y = engine.eval(data)
                val_loss.append(curr_val_loss)

                if data['batch_series'].shape[0] == 1:
                    val_pred_y = torch.tensor([val_pred_y]).to(device)

                val_outputs.append(val_pred_y)
                val_ys.append(data['batch_y'])

            s2 = time.time()
            print("Epoch: {:03d}, Validation Time: {:.4f} secs".format(i, (s2 - s1)))

            train_pre = torch.cat(train_outputs, dim=0)
            train_y = torch.cat(train_ys, dim=0)
            train_real = torch.Tensor(train_y).to(device).float()

            pred = torch.sigmoid(train_pre.squeeze()).to(device)
            real = train_real.to(device)
            train_score = metric(pred.detach(), real.detach())
            print(f'train_score: {train_score}')

            val_pre = torch.cat(val_outputs, dim=0)
            val_y = torch.cat(val_ys, dim=0)
            val_real = torch.Tensor(val_y).to(device).float()

            pred = torch.sigmoid(val_pre.squeeze()).to(device)
            real = val_real.to(device)
            val_score = metric(pred.detach(), real.detach())
            print(f'val_score: {val_score}')

            mtrain_loss = np.mean(train_loss)
            mvalid_loss = np.mean(val_loss)

            his_loss.append(mvalid_loss)
            print("-----------------------")
            print("Epoch: {:03d}, Train Loss: {:.4f}".format(i, mtrain_loss), flush=True)
            print("Epoch: {:03d}, Valid Loss: {:.4f}".format(i, mvalid_loss), flush=True)

            if mvalid_loss < best_valid_loss:
                print("###Update tasks appear###")
                best_valid_loss = mvalid_loss
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

            if epochs_since_best_mse >= args.es_patience:
                early_stopped = True
                print("Early Stop \n")
                break

        fold_train_end_time = datetime.now()
        fold_train_duration = fold_train_end_time - fold_train_start_time
        print(f"Training ends... Fold {fold_index} at {fold_train_end_time}")
        print(f"Total train time spent for fold {fold_index}: {fold_train_duration}")
        print(f"The epoch of the best result：{bestid}")
        print(f"The valid loss of the best model {str(round(his_loss[bestid - 1], 4))} \n")
        print(f"The valid metric of the best model (Amex Metric, AUC): {best_score[0]:.6g}, {best_score[1]:.6g} \n")

        fold_best_scores.append(best_score)

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

    print("================ Summary of All Folds ================")
    avg_amex = 0.0
    avg_auc = 0.0
    for idx, score in enumerate(fold_best_scores):
        print(f"Fold {idx} - Amex Metric: {score[0]:.6g}, AUC: {score[1]:.6g}")
        avg_amex += score[0]
        avg_auc += score[1]

    if len(fold_best_scores) > 0:
        avg_amex /= len(fold_best_scores)
        avg_auc /= len(fold_best_scores)
        print("------------------------------------------------------")
        print(f"Average - Metric: {avg_amex:.6g}, AUC: {avg_auc:.6g}")
    print("======================================================\n")

    return log_df


def main_test(is_predict=False):
    test_start_time = datetime.now()
    print(f"Start Testing at {test_start_time}")
    print(f'is_predict={is_predict}')

    # 动态计算所需的 emb_path 传给 Dataset
    if is_predict:
        input_path = f'../../000_data/amex/original_100pct'
        print(f'predict input_path: {input_path}')
        
        if args.emb_version and args.emb_version.startswith('v') and args.emb_version[1:].isdigit():
            v_num = args.emb_version[1:].zfill(2)
            local_emb_path = f'{input_path}/emb_{v_num}/'
            print(f'predict local_emb_path updated to: {local_emb_path}')
        else:
            local_emb_path = None
    else:
        input_path = INPUT_PATH
        test_y = pd.read_csv(f'{input_path}/test_labels.csv')['target']
        local_emb_path = emb_path  # 普通 test 使用全局原本解析好的路径

    test_series = pd.read_feather(f'{input_path}/df_nn_series_test.feather')
    test_series_idx = pd.read_feather(f'{input_path}/df_nn_series_idx_test.feather').values

    # [CHANGE] Unpack the mask and use args.seq_len dynamically
    test_series_idx, keep_mask = filter_short_seq_test_indices(test_series_idx, max_len=args.seq_len)

    if not is_predict:
        input_path = INPUT_PATH
        test_y = pd.read_csv(f'{input_path}/test_labels.csv')['target']
        # [NEW] Filter the labels so they match the predictions!
        test_y = test_y[keep_mask].values
        local_emb_path = emb_path

    # 如果当前是 Stage 1 (教师模型)，则必须启用 embedding
    use_emb_flag = True if args.stage == 1 else False
    
    # 显式传入 local_emb_path 避免全局变量污染
    test_dataset = Amex_Dataset(test_series, test_series_idx, use_embedding=use_emb_flag, emb_path=local_emb_path)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 32, # 保留原有的加速设定
        shuffle=False,
        drop_last=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=args.num_workers
    )

    models = []
    test_outputs = []

    print('Load models...')
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
            lrate=float(args.lrate),
            wdecay=args.weight_decay,
            device=device,
            epochs=args.epochs,
            stage=args.stage,
            teacher_path=None,
            is_training=False
        )

        model = engine.model
        model.load_state_dict(torch.load(model_path + f"best_model_fold_{fold_index}.pth"), strict=False)
        model.eval()
        models.append(model)

    for _, data in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            # 判断是取学生输出(索引2)还是教师输出(索引3)
            out_idx = 3 if args.stage == 1 else 2
            
            if data['batch_series'].shape[0] == 1:
                # 前向传播显式传入 stage=args.stage
                pred_y = torch.tensor([torch.stack([m(data, stage=args.stage)[out_idx] for m in models], 0).mean(0)]).to(device)
            else:
                pred_y = torch.stack([m(data, stage=args.stage)[out_idx] for m in models], 0).mean(0)

        test_outputs.append(pred_y)

    print(pred_y.shape)
    test_pre = torch.cat(test_outputs, dim=0)

    pred = torch.sigmoid(test_pre.squeeze()).to(device)

    if not is_predict:
        test_real = torch.Tensor(test_y).to(device).float()
        real = test_real.to(device)
        test_score = metric(pred, real)
        print(f'test_score: {test_score}')
    else:
        sub = test_series[['customer_ID']].iloc[test_series_idx[:, 0]].copy()
        sub['prediction'] = pred.cpu().detach().numpy()

        # Dynamically name the submission file based on seq_len
        if args.seq_len == 13:
            file_name = 'submission.csv.zip'
        else:
            file_name = f'submission_short_seq_{args.seq_len}.csv.zip'

        sub.to_csv(os.path.join(model_path, file_name), index=False, compression='zip')

    test_end_time = datetime.now()
    test_duration = test_end_time - test_start_time
    print(f"Testing ends at {test_end_time}")
    print(f"Total test time spent: {test_duration}")

    log_df = create_log_df()
    log_df['is_predict'] = [is_predict]
    if not is_predict:
        log_df['amex_metric']   = [f"{test_score[0]:.6g}"]
        log_df['AUC']           = [f"{test_score[1]:.6g}"]
        log_type = 'test'
    else:
        log_df['amex_metric']   = [None]
        log_df['AUC']           = [None]
        log_type = 'predict'
    log_df['test_start_time'] = [test_start_time.strftime('%Y-%m-%d %H:%M:%S')]
    log_df['test_end_time'] = [test_end_time.strftime('%Y-%m-%d %H:%M:%S')]
    log_df = save_log(log_type=log_type, log_df=log_df)
    return log_df


def create_log_df():
    log_df = pd.DataFrame()
    log_df['stage'] = [args.stage]
    log_df['stage_name'] = [stage_name]
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
    log_df['temperature'] = [args.temperature]
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

    if args.test:
        if args.data_type == 'original' and args.sampling == '100pct':
            print('Skip Test for orginal_100pct')
        else:
            main_test()

    if args.predict:
        log_df = main_test(is_predict=True)

    if args.submit:
        if log_df is not None:
            submit_message = log_df.to_json(orient='records')
        else:
            submit_message = f'{model_specs_template}: {model_specs}'
        os.system(
            f"""kaggle competitions submit -c amex-default-prediction -f {model_path}/submission.csv.zip -m '{submit_message}'"""
        )
        print("\n")