import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm


thin_file_path = './logs/Amex/S3_student_original_v8_12_100pct_1e-4_42/submission.csv.zip'
full_file_path = './logs/Amex/S3_student_original_v8_13_100pct_1e-4_42/submission.csv.zip'

print("Loading submissions...")
thin_sub = pd.read_csv(f'{thin_file_path}')
full_sub = pd.read_csv(f'{full_file_path}')

print("Ensembling...")
model_path = './logs/Amex/S3_student_original_v8_12and13_100pct_1e-4_42/'
submit_message = 'test ensemble thin'

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Set indices to make updating easy
full_sub.set_index('customer_ID', inplace=True)
thin_sub.set_index('customer_ID', inplace=True)

# Overwrite the predictions for the short users with your new specialized model
full_sub.update(thin_sub)

full_sub.reset_index().to_csv(f'{model_path}/final_merged_submission.csv.zip', index=False, compression='zip')

print("Submitting...")
os.system(
            f"""kaggle competitions submit -c amex-default-prediction -f {model_path}/final_merged_submission.csv.zip -m '{submit_message}'"""
        )

