import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def denoise(df):
    df['D_63'] = df['D_63'].apply(lambda t: {'CR':0, 'XZ':1, 'XM':2, 'CO':3, 'CL':4, 'XL':5}[t]).astype(np.int8)
    df['D_64'] = df['D_64'].apply(lambda t: {np.nan:-1, 'O':0, '-1':1, 'R':2, 'U':3}[t]).astype(np.int8)
    for col in tqdm(df.columns):
        if col not in ['customer_ID','S_2','D_63','D_64']:
            df[col] = np.floor(df[col]*100)
    return df

########################################
print('process train')

train = pd.read_csv('./train_data.csv')
train = denoise(train)
train.to_feather('./train.feather')

train_df_count_df = pd.DataFrame([train.shape[0]],columns=['train_df_count'])
train_df_count_df.to_feather('./train_df_count_df.feather')

del train


########################################
print('process test')

test = pd.read_csv('./test_data.csv')
test = denoise(test)
test.to_feather('./test.feather')
