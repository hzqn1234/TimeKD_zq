<div align="center">
  <h2><b> (ICDE'25) Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation </b></h2>
</div>

This repository adpats the code from TimeKD [repo](https://github.com/ChenxiLiu-HNU/TimeKD/blob/main/clm.py). Based on their code, I'm trying to adpat their method and apply on the Amex [dataset](https://www.kaggle.com/competitions/amex-default-prediction/overview), where the goal is to predict whether the credit card users will default in the future based on available data.

There are generally 3 steps to run the code:
- sbatch preprocess/run.sh
- sbatch amex_Store_1_train.sh
- sbatch amex_Store_1_test.sh
- sbatch amex_Fcst_1.sh

------------------
The Time KD source repo contains the code for their ICDE 2025 [paper](https://www.arxiv.org/abs/2505.02138), where they propose an efficient MTSF framework that leverages the calibrated language models and privileged knowledge distillation. 
