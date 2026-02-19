import warnings
warnings.simplefilter('ignore')

import pandas as pd
from tqdm import tqdm

def one_hot_encoding(df,cols,is_drop=True):
    for col in cols:
        print('one hot encoding:',col)
        dummies = pd.get_dummies(pd.Series(df[col]),prefix='oneHot_%s'%col)
        df = pd.concat([df,dummies],axis=1)
    if is_drop:
        df.drop(cols,axis=1,inplace=True)
    return df

cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]

df = pd.read_feather(f'./train.feather').append(pd.read_feather(f'./test.feather')).reset_index(drop=True)
# df = df.drop(['S_2'],axis=1)
df = one_hot_encoding(df,cat_features,True)

df_dummy_list = pd.DataFrame()
for col in tqdm(df.columns):
    if col not in ['customer_ID','S_2']:
        df[col] /= 100
        # df_dummy_list[col + '_isnull'] = df[col].isna().astype(int)
    
    df[col] = df[col].fillna(0)

# df = pd.concat([df,df_dummy_list],axis = 1)

df.to_feather('./nn_series.feather')
df.head(1).to_feather('./nn_series__sample.feather')

