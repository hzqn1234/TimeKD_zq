<div align="center">
  <h2><b> (ICDE'25) Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation </b></h2>
</div>

This repository adpats the code from TimeKD [repo](https://github.com/ChenxiLiu-HNU/TimeKD/blob/main/clm.py). Based on their code, I'm trying to adpat their method and apply on the Amex [dataset](https://www.kaggle.com/competitions/amex-default-prediction/overview), where the goal is to predict whether the credit card users will default in the future based on available data.

The Time KD source repo contains the code for their ICDE 2025 [paper](https://www.arxiv.org/abs/2505.02138), where thet propose an efficient MTSF framework that leverages the calibrated language models and privileged knowledge distillation. the rest of the page is the readme from the TimeKD repo.

<p align="center">
  <img width="900" alt="image" src="https://github.com/user-attachments/assets/c3ffa90c-604a-4ebf-bb90-cca573264c91" />
</p>

> If you find our work useful in your research. Please consider giving a star ⭐ and citation 📚:

```bibtex
@inproceedings{liu2025timekd,
  title={Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation},
  author={Chenxi Liu and Hao Miao and Qianxiong Xu and Shaowen Zhou and Cheng Long and Yan Zhao and Ziyue Li and Rui Zhao},
  booktitle = {ICDE},
  year={2025}
```

## Dependencies

* Python 3.10
* PyTorch 2.1.2
* cuda 12.1
* torchvision 0.16.2

```bash
> conda env create -f env_ts.yaml
```

## Datasets
Datasets can be obtained from [TimesNet](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2)
)
## Usages
* ### Storage 

```bash
chmod +x Store.sh
./Store.sh
```

* ### Forecasting
   
```bash
chmod +x Fcst.sh
./Fcst.sh
```

* ### Standard deviation of the MAE and RMSE
<p align="center">
  <img width="453" alt="image" src="https://github.com/user-attachments/assets/dd1110ab-3996-4a47-a727-7f33dbac2fd1" />
</p>

We have conducted experiments on ETTm1, ETTm2, ETTh1, ETTh2, Weather, and Exchange with three different random seeds (i.e., 2024, 6666, 8888) and report the Mean and Standard deviation (Std) of the MSE and MAE in Table A.
We observe that the standard deviation of MSE and MAE is minimal ranging from 0.0014 to 0.0042. These results indicate that TimeKD exhibits strong robustness and stability across different random seeds.

## Contact Us
For inquiries or further assistance, contact us at [chenxi.liu@ntu.edu.sg](mailto:chenxi.liu@ntu.edu.sg).
