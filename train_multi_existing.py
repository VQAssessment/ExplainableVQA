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

## You need to install DOVER
from dover import datasets
from dover import DOVER

import wandb
import argparse

from model import TextEncoder, MaxVQA, EnhancedVisualEncoder


import time

def rescale(x):
    x = np.array(x)
    x = (x - x.mean()) / x.std()
    return x #1 / (1 + np.exp(-x))

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
    return ((loss0 + loss1) / 2).float()

def count_parameters(model):
    for name, module in model.named_children():
        print(name, "|", sum(p.numel() for p in module.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class MixVisualFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, visual_features: dict, gts: dict, indices: dict, train_length=800):
        super().__init__()
        self.visual_features, self.gts = {}, {}
        for key in visual_features:
            self.visual_features[key] = [visual_features[key][ind] for ind in indices[key]]
            self.gts[key] = rescale([gts[key][ind] for ind in indices[key]])
        
        self.train_length = train_length
    def __getitem__(self, index):
        mix_feats = []
        mix_gts = []
        for key in self.gts:
            kidx = random.randrange(len(self.gts[key]))
            mix_feats.append(self.visual_features[key][kidx])
            mix_gts.append(self.gts[key][kidx])
        return mix_feats, mix_gts
    def __len__(self):
        return self.train_length
    
    
class MaxVisualFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, visual_features, max_gts, indices=None):
        super().__init__()
        if indices == None:
            indices = range(len(visual_features))
            print("Using all indices:", indices)
        self.visual_features = [visual_features[ind] for ind in indices]
        self.gts = [max_gts[ind] for ind in indices]
        
        
    def __getitem__(self, index):
        return self.visual_features[index], self.gts[index]
    def __len__(self):
        return len(self.gts)
    
def encode_text_prompts(prompts,device="cuda"):
        text_tokens = tokenizer(prompts).to(device)
        with torch.no_grad():
            embedding = model.token_embedding(text_tokens)
            text_features = model.encode_text(text_tokens).float()
        return text_tokens, embedding, text_features


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--opt",
        type=str,
        default="./LKY.yml",
        help="the option file",
    )
    
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda",
        help="the option file",
    )
    
    args = parser.parse_args()

    device = args.device

    ## initialize datasets

    with open(args.opt, "r") as f:
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



    num_datasets = len(val_datasets)
    ## encode initialized prompts 

    context = "X"

    positive_descs = ["high quality"] * (num_datasets+1)

    negative_descs = ["low quality"] * (num_datasets+1)

    pos_prompts = [ f"a {context} {desc} photo" for desc in positive_descs]
    neg_prompts = [ f"a {context} {desc} photo" for desc in negative_descs]

    tokenizer = open_clip.get_tokenizer("RN50")

    text_tokens, embedding, text_feats = encode_text_prompts(pos_prompts + neg_prompts, device=device)

    ## Load model
    text_encoder = TextEncoder(model).to(device)
    visual_encoder = EnhancedVisualEncoder(model, fast_vqa_encoder).to(device)



    ### Extract Features before training

    gts, paths = {}, {}

    for val_name, val_dataset in val_datasets.items():
        gts[val_name] = [val_dataset.video_infos[i]["label"] for i in range(len(val_dataset))]

    for val_name, val_dataset in val_datasets.items():
        paths[val_name] = [val_dataset.video_infos[i]["filename"] for i in range(len(val_dataset))]

    val_prs = {}

    feats = {}

    print("Extracting pooled features...")


    os.makedirs("features",exist_ok=True)
    for val_name, val_dataset in val_datasets.items():
        feat_path = f"features/maxvqa_vis_{val_name}.pkl"
        if glob.glob(feat_path):
            print("Found pre-extracted visual features...")

            s = time.time()
            if "maxwell" in val_name:
                feats[val_name] = [f.float() for f in torch.load(feat_path)]
            else:
                with open(feat_path, "rb") as f:
                    feats[val_name] = pkl.load(f)
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
                    feats[val_name].append(vis_feats.mean((-3,-2),keepdim=True).cpu().numpy())
                torch.cuda.empty_cache()

            with open(feat_path, "wb") as f:
                pkl.dump(feats[val_name], f)

    print("Training Starts")




    all_srccs, all_plccs, all_s_srccs, all_s_plccs = [], [], [], []
    for split in range(10):

        run = wandb.init(
            project="MaxVQA",
            name=f"mixvqa_{split}_LKY",
            reinit=True,
            settings=wandb.Settings(start_method="thread"),
        )

        best_metric = -1
        train_dataloaders, test_dataloaders = {}, {}
        train_inds = {}
        print(f"Mix-dataset training in split {split}:")

        maxvqa = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)
        print(f'The model has {count_parameters(maxvqa):,} trainable parameters')
        optimizer = torch.optim.AdamW(maxvqa.parameters(),lr=1e-3)
        train_feats = {}
        for val_name in feats:
            
            if val_name == "val-maxwell":
                test_dataset = MaxVisualFeatureDataset(feats[val_name], gts[val_name])
                test_dataloaders[val_name] = torch.utils.data.DataLoader(test_dataset, batch_size=16)
                continue
                
            train_feats[val_name] = feats[val_name]
            
            if val_name == "train-maxwell":
                train_inds[val_name] = list(range(len(gts[val_name])))
                continue
            
            random.seed((split+1)*42)
            train_ind = random.sample(range(len(gts[val_name])), int(0.8 * len(gts[val_name])))
            train_inds[val_name] = train_ind
            val_ind = [ind for ind in range(len(gts[val_name])) if ind not in train_ind]
            #train_dataset = MaxVisualFeatureDataset(feats[val_name], gts[val_name], train_ind)
            #train_dataloaders[val_name] = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

            test_dataset = MaxVisualFeatureDataset(feats[val_name], gts[val_name], val_ind)
            test_dataloaders[val_name] = torch.utils.data.DataLoader(test_dataset, batch_size=16)

        train_dataset = MixVisualFeatureDataset(train_feats, gts, train_inds)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

        maxvqa_ema = MaxVQA(text_tokens, embedding, text_encoder, share_ctx=True).to(device)

        for epoch in (range(20)):
            print(f"Split {split}, Epoch {epoch}")
            maxvqa.train()
            for data in tqdm(train_dataloader,desc="Training"):
                    optimizer.zero_grad()
                    mix_vis_feat, mix_gt = data
                    for i, (vis_feat, gt) in enumerate(zip(mix_vis_feat, mix_gt)):
                        res = maxvqa(vis_feat.cuda(), text_encoder)
                        loss = plcc_loss(res[...,0,i], gt.cuda().float())
                        loss.backward()
                        optimizer.step()
                    res = maxvqa(torch.cat(mix_vis_feat,0).cuda(), text_encoder)
                    loss = plcc_loss(res[...,0,-1], torch.cat(mix_gt, 0).cuda().float())
                    loss.backward()
                    optimizer.step()

                    model_params = dict(maxvqa.named_parameters())
                    model_ema_params = dict(maxvqa_ema.named_parameters())
                    for k in model_params.keys():
                        model_ema_params[k].data.mul_(0.999).add_(
                            model_params[k].data, alpha=1 - 0.999)

            maxvqa.eval()

            metric = 0
            srccs, plccs = np.zeros(num_datasets), np.zeros(num_datasets)
            shared_srccs, shared_plccs = np.zeros(num_datasets), np.zeros(num_datasets)
            for i, (val_name, test_dataloader) in enumerate(test_dataloaders.items()):
                val_sprs, val_prs, val_gts = [], [], []
                for data in tqdm(test_dataloader, desc=val_name):
                    with torch.no_grad():
                        vis_feat, gt = data
                        res_s = maxvqa_ema(vis_feat.cuda(), text_encoder)
                        val_sprs.extend(list(res_s[...,0,i].cpu().numpy()))
                        val_prs.extend(list(res_s[...,0,-1].cpu().numpy()))
                        val_gts.extend(list(gt.cpu().numpy()))

                #val_sprs = np.stack(val_sprs, 0)
                val_prs = np.stack(val_prs, 0)
                val_gts = np.stack(val_gts, 0)

                shared_srcc, shared_plcc = spearmanr(val_prs,val_gts)[0], pearsonr(val_prs,val_gts)[0]
                print("Shared", val_name,shared_srcc,shared_plcc)
                wandb.log({f"SRCC_{val_name}": shared_srcc, f"PLCC_{val_name}": shared_plcc})
                
                shared_srccs[i] = shared_srcc
                shared_plccs[i] = shared_plcc

                srcc, plcc = spearmanr(val_sprs,val_gts)[0], pearsonr(val_sprs,val_gts)[0]
                print("Specific", val_name,srcc,plcc)
                wandb.log({f"SRCC_s_{val_name}": srcc, f"PLCC_s_{val_name}": plcc})
                metric += srcc + plcc + shared_plcc + shared_srcc



                srccs[i] = srcc
                plccs[i] = plcc


            if metric > best_metric:
                best_metric = metric
                best_srccs = srccs
                best_plccs = plccs
                best_shared_srccs = shared_srccs
                best_shared_plccs = shared_plccs
                torch.save(maxvqa_ema.state_dict(), f"mixvqa_split_{split}.pt")

        all_srccs.append(best_srccs)
        all_plccs.append(best_plccs)
        
        all_s_srccs.append(best_shared_srccs)
        all_s_plccs.append(best_shared_plccs)

    print(f"SRCC: {list(val_datasets.keys())}", sum(all_srccs) / 10)
    print(f"PLCC: {list(val_datasets.keys())}", sum(all_plccs) / 10)

    
    print(f"Shared SRCC: {list(val_datasets.keys())}", sum(all_s_srccs) / 10)
    print(f"Shared PLCC: {list(val_datasets.keys())}", sum(all_s_plccs) / 10)
