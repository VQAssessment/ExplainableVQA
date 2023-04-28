import os
import glob

import argparse
import pickle as pkl
import random

import open_clip
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kendalltau as kendallr
from tqdm import tqdm

import time


def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float() #+ 0.3 * rank_loss(y_pred[...,None], y[...,None])

def count_parameters(model):
    for name, module in model.named_children():
        print(name, "|", sum(p.numel() for p in module.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MaxVisualFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, visual_features, max_gts, indices=None):
        super().__init__()
        if indices == None:
            indices = range(len(visual_features))
            print("Using all indices:", indices)
        self.visual_features = [visual_features[ind] for ind in indices]
        self.gts = [max_gts.iloc[ind].values for ind in indices]
        
    def __getitem__(self, index):
        return self.visual_features[index], torch.Tensor(self.gts[index])
    def __len__(self):
        return len(self.gts)
    
def encode_text_prompts(prompts,device="cuda"):
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            embedding = model.token_embedding(text_tokens)
            text_features = model.encode_text(text_tokens).float()
        return text_tokens, embedding, text_features

## You need to install DOVER
from dover import datasets
from dover import DOVER

import wandb

from model import TextEncoder, MaxVQA, EnhancedVisualEncoder

device = "cuda"

## initialize datasets

with open("maxvqa.yml", "r") as f:
    opt = yaml.safe_load(f)   
    
val_datasets = {}
for name, dataset in opt["data"].items():
    val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])
    
    
## initialize clip

print(open_clip.list_pretrained())
model, _, _ = open_clip.create_model_and_transforms("RN50",pretrained="openai")
model = model.to(device)

## initialize fast-vqa encoder

fast_vqa_encoder = DOVER(**opt["model"]["args"]).to(device)
fast_vqa_encoder.load_state_dict(torch.load("../DOVER/pretrained_weights/DOVER.pth"),strict=False)


## encode initialized prompts 

context = "X"

positive_descs = ["high quality", "good content", "organized composition", "vibrant color", "contrastive lighting", "consistent trajectory",
                  "good aesthetics",
                  "sharp", "in-focus", "noiseless", "clear-motion", "stable", "well-exposed", 
                  "original", "fluent", "clear", 
                 ]


negative_descs = ["low quality", "bad content", "chaotic composition", "faded color", "gloomy lighting", "incoherent trajectory",
                  "bad aesthetics",
                  "fuzzy", "out-of-focus", "noisy", "blurry-motion", "shaky", "poorly-exposed",
                  "compressed", "choppy", "severely degraded", 
                 ]

         
pos_prompts = [ f"a {context} {desc} photo" for desc in positive_descs]
neg_prompts = [ f"a {context} {desc} photo" for desc in negative_descs]

tokenizer = open_clip.get_tokenizer("RN50")

text_tokens, embedding, text_feats = encode_text_prompts(pos_prompts + neg_prompts, device=device)

## Load model
text_encoder = TextEncoder(model).to(device)
visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder).to(device)

maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)



### Extract Features before training

gts, paths = {}, {}

for val_name, val_dataset in val_datasets.items():
    gts[val_name] = [val_dataset.video_infos[i]["label"] for i in range(len(val_dataset))]
    
for val_name, val_dataset in val_datasets.items():
    paths[val_name] = [val_dataset.video_infos[i]["filename"] for i in range(len(val_dataset))]

val_prs = {}

feats = {}

print("Extracting features...")


print(val_datasets.keys())



os.makedirs("features",exist_ok=True)

for val_name, val_dataset in val_datasets.items():
    if "maxwell" not in val_name:
        print(f"Omitting {val_name}")
        continue
    feat_path = f"features/maxvqa_vis_{val_name}.pkl"
    if glob.glob(feat_path):
        print("Found pre-extracted visual features...")
        s = time.time()
        feats[val_name] = torch.load(feat_path)
        print(f"Successfully loaded {val_name}, elapsed {time.time() - s:.2f}s.")
    else:
        print("Extracting on-the-fly...")
        feats[val_name] = []
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=8, pin_memory=True,
        )
        for i, data in enumerate(tqdm(val_loader, desc=f"Extracting in dataset [{val_name}].")):
            with torch.no_grad():
                vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
                feats[val_name].append(vis_feats.half().cpu())
            torch.cuda.empty_cache()

        torch.save(feats[val_name], feat_path)
            
print("Training Starts")

import pandas as pd
max_gts_train = pd.read_csv("MaxWell_train.csv")
max_gts_val = pd.read_csv("MaxWell_val.csv")


print(f'The model has {count_parameters(maxvqa):,} trainable parameters')
optimizer = torch.optim.AdamW(maxvqa.parameters(),lr=1e-3)

train_dataset = MaxVisualFeatureDataset(feats["train-maxwell"], max_gts_train)
train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = MaxVisualFeatureDataset(feats["val-maxwell"], max_gts_val)
test_dataloader =  torch.utils.data.DataLoader(test_dataset, batch_size=16)

#state_dict = torch.load("maxvqa.pt")
maxvqa_ema = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

if True: 
    val_prs, val_gts = [], []
    for data in tqdm(test_dataloader):
        with torch.no_grad():
            vis_feat, gt = data
            res = maxvqa(vis_feat.cuda(), text_encoder)[:,0]
            val_prs.extend(list(res.cpu().numpy()))
            val_gts.extend(list(gt.cpu().numpy()))
    val_prs = np.stack(val_prs, 0)
    val_gts = np.stack(val_gts, 0)
    
    
        
    for i, key in zip(range(16), max_gts_train):
        srcc, plcc = spearmanr(val_prs[:,i],val_gts[:,i])[0], pearsonr(val_prs[:,i],val_gts[:,i])[0]
        print(key,srcc,plcc)

best_all_plcc = 0

run = wandb.init(
            project="MaxVQA",
            name=f"maxvqa_maxwell_pushed",
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
)

for epoch in (range(20)):
    print(epoch)
    maxvqa.train()
    for data in tqdm(train_dataloader):
        optimizer.zero_grad()
        vis_feat, gt = data
        res = maxvqa(vis_feat.cuda(), text_encoder)
        loss, aux_loss = 0, 0
        for i in range(16):
            loss += plcc_loss(res[:,0,i], gt[:,i].cuda().float())
            for j in range(i+1,16):
                aux_loss += 0.005 * (0.5-plcc_loss(res[:,0,i], res[0,:,j]))
                
        wandb.log({"loss": loss.item(), "aux_loss": aux_loss.item()})
        loss += aux_loss
        loss.backward()
        optimizer.step()
                    
        model_params = dict(maxvqa.named_parameters())
        model_ema_params = dict(maxvqa_ema.named_parameters())
        for k in model_params.keys():
            model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999)
                
    maxvqa.eval()

    val_prs, val_gts = [], []
    for data in tqdm(test_dataloader):
        with torch.no_grad():
            vis_feat, gt = data
            res = maxvqa_ema(vis_feat.cuda(), text_encoder)[:,0]
            val_prs.extend(list(res.cpu().numpy()))
            val_gts.extend(list(gt.cpu().numpy()))
    val_prs = np.stack(val_prs, 0)
    val_gts = np.stack(val_gts, 0)
    
    
    all_plcc = 0    
    for i, key in zip(range(16), max_gts_train):
        srcc, plcc = spearmanr(val_prs[:,i],val_gts[:,i])[0], pearsonr(val_prs[:,i],val_gts[:,i])[0]
        print(key,srcc,plcc)
        all_plcc += plcc
    
    if all_plcc > best_all_plcc:
        with open("maxvqa_validation_results.pkl","wb") as f:
            pkl.dump(val_prs, f)
        best_all_plcc = all_plcc
        torch.save(maxvqa_ema.state_dict(), "maxvqa_pushed_away_maxwell.pt")
        