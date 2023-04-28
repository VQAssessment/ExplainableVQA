# Towards Explainable In-the-Wild Video Quality Assessment

A subjective study (Maxwell database) and a language-prompt approach (MaxVQA Model).
The database (training part) will be released later.

## Installation

Install and modify OpenCLIP:

```
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
sed -i '92s/return x\[0\]/return x/' src/open_clip/modified_resnet.py 
pip install -e .
```

Install DOVER for Pre-processing and FAST-VQA weights:

```
git clone https://github.com/vqassessment/DOVER.git
cd DOVER
pip install -e .
mkdir pretrained_weights 
cd pretrained_weights 
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth 
```

## Huggingface Workspace on MaxVQA


## Development on MaxVQA

### Inference from Videos

```python
infer_from_videos.py
```

### Inference from Pre-extracted Features

```python
infer_from_feats.py
```

For the first time, the script will extract features from videos.


### Training on Mixed Existing VQA Databases

For the default setting, train on LIVE-VQC, KoNViD-1k, and YouTube-UGC.

```python
train_multi_existing.py -o LKY.yml
```

You can also modify the yaml file to include more datasets for training.

## Obtaining Data for the Maxwell Database

TBA.

## Citation

TBA.