import os
import time

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
text_encoder = TextEncoder(model)
visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder)
maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).cuda()

state_dict = torch.load("maxvqa_maxwell.pt")
maxvqa.load_state_dict(state_dict)
maxvqa.initialize_inference(text_encoder)

### evaluation

gts, paths = {}, {}

for val_name, val_dataset in val_datasets.items():
    gts[val_name] = [val_dataset.video_infos[i]["label"] for i in range(len(val_dataset))]
    
for val_name, val_dataset in val_datasets.items():
    paths[val_name] = [val_dataset.video_infos[i]["filename"] for i in range(len(val_dataset))]

val_prs = {}

for val_name, val_dataset in val_datasets.items():
    if "train" in val_name:
        continue
    val_prs[val_name] = []
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=6, pin_memory=True,
    )
    for i, data in enumerate(tqdm(val_loader, desc=f"Evaluating in dataset [{val_name}].")):
        with torch.no_grad():
            vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
            res = maxvqa(vis_feats, text_encoder, train=False)
            val_prs[val_name].extend(list(res.cpu().numpy()))
        torch.cuda.empty_cache()
    val_gts = gts[val_name]
    if val_name != "val-maxwell":
        for i in range(16):
            print(f"Generalization Evaluating: {positive_descs[i]}<->{negative_descs[i]}", pearsonr([pr[i] for pr in val_prs[val_name]], gts[val_name])[0])
    else:
        import pandas as pd
        max_gts = pd.read_csv("MaxWell_val.csv")
        for key, i in zip(max_gts, range(16)):
            print(f"Evaluating {key}: {positive_descs[i]}<->{negative_descs[i]}", pearsonr([pr[i] for pr in val_prs[val_name]], max_gts[key])[0])
        
with open("maxvqa_global_results.pkl","wb") as f:
    pkl.dump(val_prs, f)