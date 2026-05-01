import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm

# /export/home2/zongqi001/007_amex_rank1/lgbm/output/LGB_with_manual_feature/__0.80792__submission.csv.zip
# /export/home2/zongqi001/007_amex_rank1/lgbm/output/LGB_with_manual_feature_and_series_oof/__0.80784__submission.csv.zip

# p0_file_path = '../../005_best/ensemble__0.80987/__0.80824__lgb_manual__submission.csv.zip'
# p1_file_path = '../../005_best/ensemble__0.80987/__0.80866__lgb_fe__submission.csv.zip'

# p2_file_path = './logs/Amex/S2_distill_original_v8_100pct_1e-4_42/submission.csv.zip'
# p2_file_path = '/export/home2/zongqi001/004_TimeKD/TimeKD_02/logs/Amex/S3_student_original_v8_AllMerged_100pct_1e-4_42/submission.csv.zip'
# p2_file_path = './logs/Amex/S3_student_original_v8_10and13_100pct_1e-4_42/final_merged_submission.csv.zip'

p0_file_path = '../../007_amex_rank1/lgbm/output/LGB_with_manual_feature/__0.80792__submission.csv.zip'
p1_file_path = '../../007_amex_rank1/lgbm/output/LGB_with_manual_feature_and_series_oof/__0.80784__submission.csv.zip'
p2_file_path = '../../005_best/ensemble__0.80987/__0.80141__tsf_series__submission.csv.zip'
p3_file_path = '../../005_best/ensemble__0.80987/__0.80558__tsf_fe__submission.csv.zip'

# os.system(f"""kaggle competitions submit -c amex-default-prediction -f {p0_file_path} -m 'last time best ensemble - p0'""")
# os.system(f"""kaggle competitions submit -c amex-default-prediction -f {p1_file_path} -m 'last time best ensemble - p1'""")
# os.system(f"""kaggle competitions submit -c amex-default-prediction -f {p2_file_path} -m '0.80178 - p2'""")
# os.system(f"""kaggle competitions submit -c amex-default-prediction -f {p3_file_path} -m 'last time best ensemble - p3'""")

print("Loading submissions...")
p0 = pd.read_csv(f'{p0_file_path}')
p1 = pd.read_csv(f'{p1_file_path}')
p2 = pd.read_csv(f'{p2_file_path}')
p3 = pd.read_csv(f'{p3_file_path}')

print("Ensembling...")
model_path = './output'
submit_message = 'test ensemble'

p0['prediction'] = p0['prediction']*0.3 + p1['prediction']*0.35 + p2['prediction']*0.15 + p3['prediction']*0.1
# p0['prediction'] = p0['prediction']*0.25 + p1['prediction']*0.45 + p2['prediction']*0.15 + p3['prediction']*0.15
p0.to_csv(f'{model_path}/final_submission.csv.zip',index=False, compression='zip')

print("Submitting...")
os.system(
            f"""kaggle competitions submit -c amex-default-prediction -f {model_path}/final_submission.csv.zip -m '{submit_message}'"""
        )

