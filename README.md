# IMU-Video-MAE

(ECCV2024) Masked Video and Body-worn IMU Autoencoder for Egocentric Action Recognition [Paper](https://arxiv.org/pdf/2407.06628)

<img src="./evi-mae.png" />

## Installation

Requirements:
- PyTorch
- DGL (https://www.dgl.ai/pages/start.html)
- decord
- timm
- einops
- scikit-learn
- pandas

Example:
```bash
conda create -n evi-mae python=3.11
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu118/repo.html
pip install decord timm==0.4.5 einops scikit-learn pandas
```

## Data Preparation
We use the [CMU-MMAC](http://kitchen.cs.cmu.edu) dataset and the [WEAR](https://github.com/mariusbock/wear) dataset.

For the CMU-MMAC dataset, we follow [mmac_captions](https://github.com/hitachi-rd-cv/mmac_captions) to preprocess the data and leverage the action labels from [EgoProceL](https://github.com/Sid2697/EgoProceL-egocentric-procedure-learning/blob/main/EgoProceL-download-README.md).

We repack the two datasets which can be downloaded at [CMU-MMAC](https://drive.google.com/file/d/1H0zCzvEeDIAgT9FLfcSoZp3FXP42ibBP/view?usp=share_link) and [WEAR](https://drive.google.com/file/d/1dQCWDOhg70-s7ErP4FBVBKaTHREUkdKo/view?usp=share_link).

## Checkpoints

Before pretraining or fine-tuning, we create an adapted [VideoMAE](https://arxiv.org/abs/2203.12602) checkpoint to initialize our model. Please download it at [small-video-mae](https://drive.google.com/file/d/1JxQtmgoxIxqFdY-3CAjUlZvs2JTn_l3A/view?usp=share_link). The adaptation method follows [cav-mae](https://github.com/YuanGongND/cav-mae/tree/master?tab=readme-ov-file#adapt-vision-mae-checkpoint).

We also provide our pretrained and fine-tuned checkpoints at [share_link](https://drive.google.com/file/d/1N0U-PR8ydHx-BtWz_v1QUrCNkZGWQ1KV/view?usp=share_link).

## MAE Pretraining

Please finish the data preparation and checkpoints preparation first, and then organize the data and checkpoints in the following structure:

```
data_release/
    cmu-mmac-release/
    wear-release/
    videomae_adapt_ckpt/
```

Then, pretraining can be conducted by:

`cd egs/release; bash pretrain_cmummac.sh`

`cd egs/release; bash pretrain_wear.sh`

In the shell scripts, `$dataset_base_path` is the path to the `data_release` folder.

## Fine-tuning for Action Recognition

`cd egs/release; bash finetune_cmummac.sh`

`cd egs/release; bash finetune_wear.sh`

In the shell scripts, please modify the `dataset_base_path` and `pretrain_path`.

## Training for Action Recognition without MAE Pretraining

Our model can be trained without MAE pretraining. You can optionally use the adapted VideoMAE checkpoint to initialize the model or train from scratch.

`cd egs/release; bash train_cmummac.sh`

`cd egs/release; bash train_wear.sh`

## Training for Action Recognition with 4 IMUs (no video) without MAE Pretraining

Our model can be trained without video.

`cd egs/release; bash train_cmummac_imuonly.sh`

`cd egs/release; bash train_wear_imuonly.sh`

## Training for Action Recognition with 1 IMU

The easiest way to adapt our model to one-IMU setting is to use `train_cmummac_imuonly.sh` and set `imu_enable_graph` to `False`. Then, make 4 copies of the one input IMU in the dataloader.

## Reference

Our code implementation is based on the following repositories:

https://github.com/YuanGongND/cav-mae

https://github.com/THUDM/GraphMAE

https://github.com/MCG-NJU/VideoMAE

## Citation

```
@inproceedings{zhang2025masked,
  title={Masked video and body-worn IMU autoencoder for egocentric action recognition},
  author={Zhang, Mingfang and Huang, Yifei and Liu, Ruicong and Sato, Yoichi},
  booktitle={European Conference on Computer Vision},
  pages={312--330},
  year={2025},
  organization={Springer}
}
```
