import torch
import torch.nn as nn

class EnhancedVisualEncoder(nn.Module):
    def __init__(self, clip_model, fast_vqa_encoder):
        super().__init__()
        self.clip_visual = clip_model.visual
        self.fast_vqa_encoder = fast_vqa_encoder.technical_backbone
        
    def forward(self, x_aes, x_tech):
        
        # frame-wise
        x_aes = x_aes.transpose(1,2).reshape(-1,3,224,224)
        clip_feats = self.clip_visual(x_aes)
        clip_feats = clip_feats[1:].reshape(7,7,-1,1024).permute(3,2,0,1)
        clip_feats = clip_feats.reshape(1024, -1, 64, 49).permute(1,2,3,0)

        # chunk-wise
        x_tech = x_tech.reshape(-1,3,4,32,224,224).permute(0,2,1,3,4,5).reshape(-1,3,32,224,224)
        fast_feats = self.fast_vqa_encoder(x_tech).reshape(-1,4,768,16,7,7).permute(0,1,3,4,5,2)
        fast_feats = fast_feats.reshape(-1,64,49,768)
        return torch.cat((clip_feats, fast_feats), -1)
        