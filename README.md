# Official implementation of paper: Revisiting Continuity of Image Tokens for Cross-Domain Few-shot Learning(ICML 2025 Spotlight)

# 1. About this code

This code is for the paper: Revisiting Continuity of Image Tokens for Cross-Domain Few-shot Learning(ICML 2025 Spotlight)

# 2. Setup and datasets

## 2.1. Setup

An Anaconda environment is recommended:

```
conda create --name py36 python=3.6
conda activate py36
conda install pytorch torchvision -c pytorch
pip3 install scipy>=1.3.2
pip3 install tensorboardX>=1.4
pip3 install h5py>=2.9.0
```

## 2.2. Datasets

Five datasets, including miniImagenet, CropDiseases, EuroSAT, ISIC2018, and ChestX, are used.

Following the [FWT-repo](https://github.com/hytseng0509/CrossDomainFewShot) to download and set up all datasets.

Remember to modify your dataset dir in the 'options.py'

# 3. Usage

## 3.1. Training

```
python network_train.py --stage pretrain --name ReCIT --model VIT_S  --save_freq 1 --stop_epoch 50 --optimizer adamW --decay 0.01 --n_shot 5 --warmup_endingepoch 1 --train_aug
```

## 3.2. Testing

```
#test target dataset, e.g., ISIC
python network_test.py --ckp_path output/checkpoints/ReCIT/best_ave_model.tar  --model VIT_S --stage pretrain --dataset ISIC --n_shot 5 
```

The training script also includes a test for each epoch.

## 3.3. Transductive evaluation

In method/protonet.py, there are commented codes for the transductive evaluation, which you can uncomment to unlock the feature.

# 4. Note

Notably, our code is built upon the Meta-FDMixup: Cross-Domain Few-Shot Learning Guided by Labeled Target Data. (ACM MM 2021)
