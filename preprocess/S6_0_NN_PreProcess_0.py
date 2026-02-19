import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm

print("S6_0_NN_PreProcess_0...")

df_full =  pd.read_feather('./nn_series.feather')
df_full['idx'] = df_full.index
y = pd.read_csv('./train_labels.csv')

train_df_count_df = pd.read_feather('./train_df_count_df.feather')
train_df_count    = train_df_count_df.values[0][0]

df_nn_series_train = df_full[:train_df_count].reset_index(drop=True)
df_nn_series_test = df_full[train_df_count:].reset_index(drop=True)

df_nn_series_train['idx'] = df_nn_series_train.index
df_nn_series_test['idx'] = df_nn_series_test.index


df_nn_series_idx_train = df_nn_series_train.groupby('customer_ID',sort=False).idx.agg(['min','max']).reset_index(drop=True)
df_nn_series_idx_train['feature_idx'] = np.arange(len(df_nn_series_idx_train))
df_nn_series_train = df_nn_series_train.drop(['idx'],axis=1)

df_nn_series_idx_test = df_nn_series_test.groupby('customer_ID',sort=False).idx.agg(['min','max']).reset_index(drop=True)
df_nn_series_idx_test['feature_idx'] = np.arange(len(df_nn_series_idx_test))
df_nn_series_test = df_nn_series_test.drop(['idx'],axis=1)

df_nn_series_train    .to_feather('./df_nn_series_train.feather')
df_nn_series_idx_train.to_feather('./df_nn_series_idx_train.feather')

df_nn_series_test     .to_feather('./df_nn_series_test.feather')
df_nn_series_idx_test .to_feather('./df_nn_series_idx_test.feather')

