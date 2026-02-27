print("Starting amex_store_emb.py...")
import os
# [关键] 防止 Tokenizer 并行导致的 CPU 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Environment set: TOKENIZERS_PARALLELISM = false")

import torch
import time
import math
import h5py
import argparse
import warnings # [新增] 导入 warnings 模块用于触发截断警告
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from amex_generate_embedding import GenPromptEmb
from tqdm import tqdm

import pandas as pd
import numpy as np
from datetime import datetime

print('ok1')
import torch
import torch.nn as nn

print('ok2')
