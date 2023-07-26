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

from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition




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
    

    dopt = opt["data"]["val-ytugc"]["args"]

    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            # resized temporal sampling for TQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            # temporal sampling for AQE in DOVER
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )
    
    
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

title = """<h1 align="center">Demo of MaxVQA</h1>"""


dim_means = [0.64816874, 0.7389499, 0.72287375, 0.7463802 , 0.8296112 ,
       0.7190873 , 0.6247339 , 0.54232854, 0.629289  , 0.5732389 ,
       0.69551915, 0.49694163, 0.68725526, 0.60615003, 0.88730884,
       0.5536831]

dim_stds = [0.16527903, 0.1164713 , 0.12123751, 0.13628706, 0.12584831,
       0.11652958, 0.15417372, 0.18802027, 0.16696042, 0.15306316,
       0.153056  , 0.1480174 , 0.1273537 , 0.16313092, 0.07106569,
       0.1876018]

def rescale(raw_score, dim=-1):
    ## Not Fully Implemented Yet
    x = (raw_score - dim_means[dim]) / dim_stds[dim]
    return 1 / (1 + np.exp(-x))

def format_maxvqa(output):
    output_ = ""
    output = list(output[0])
    for i, (pos, neg, score) in enumerate(zip(positive_descs, negative_descs, output)):
        output_ += f"Axis {pos}<-->{neg} Score: {rescale(score, i)*100:.1f}\n"
    return output_

### evaluation

import gradio as gr

mean = torch.FloatTensor([123.675, 116.28, 103.53]).reshape(-1,1,1,1)
std = torch.FloatTensor([58.395, 57.12, 57.375]).reshape(-1,1,1,1)

def inference(video):
    ## Your custom video preprocessing here
    with torch.no_grad():
        print("Path:", video)
        video_data, _ = spatial_temporal_view_decomposition(
            video, dopt["sample_types"], temporal_samplers,
        )
        try:
            print(video_data["aesthetic"].shape, video_data["aesthetic"].dtype)
            print(video_data["technical"].shape, video_data["technical"].dtype)
        except:
            pass
        # Assuming that video_data is the preprocessed video from above step
        data = {"aesthetic": (video_data["aesthetic"] - mean ) / std,
                "technical": (video_data["technical"] - mean ) / std}

        vis_feats = visual_encoder(data["aesthetic"].to(device), data["technical"].to(device))
        res = maxvqa(vis_feats, text_encoder, train=False)
        output = list(res.cpu().numpy()) # Modify this part if your output is not a numpy array
        return format_maxvqa(output)

iface = gr.Interface(fn=inference, 
                     inputs=gr.inputs.Video(source="upload"),
                     outputs="text")

gr.Markdown(title)

iface.launch(share=True)
