import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_ln = nn.Linear(in_channels, hidden_channels, bias=False)
        self.out_ln = nn.Linear(hidden_channels, out_channels, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        return self.out_ln(self.dropout(self.gelu(self.in_ln(x))))
    
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.attn_mask = clip_model.attn_mask

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    
class MaxVQA(nn.Module):
    """
        Modified CLIP, which combined prompt tuning and feature adaptation.
        The spatial and temporal naturalnesses are fed as final features.
        Implcit features is also optional fed into the model.
    """
    def __init__(self, text_tokens, embedding, text_encoder, n_ctx=1, share_ctx=False):
        
        super().__init__()
        self.device = "cuda"
        self.implicit_mlp = MLP(1792,64,1025)
        self.tokenized_prompts = text_tokens
        #self.text_encoder = TextEncoder(clip_model)
        self.share_ctx = share_ctx
        
        if n_ctx > 0:
            if not share_ctx:
                self.ctx = nn.Parameter(embedding[:, 1:1+n_ctx].clone())
            else:
                self.ctx = nn.Parameter(embedding[0:1, 1:1+n_ctx].clone())
        else:
            self.register_buffer("ctx", embedding[:, 1:1, :])
            print("Disabled Context Prompt")
        self.register_buffer("prefix", embedding[:, :1, :].clone())  # SOS
        self.register_buffer("suffix", embedding[:, 1 + n_ctx:, :].clone())# CLS, EOS
        
        self.prefix.requires_grad = False
        self.suffix.requires_grad = False
        self.dropout = nn.Dropout(0.5)

        
        
        n_prompts = self.get_text_prompts()
        self.text_feats = text_encoder(n_prompts.cuda(), self.tokenized_prompts)
        
    def get_text_prompts(self):
        if self.share_ctx:
            return torch.cat(
                [
                    self.prefix,  # (n_cls, 1, dim)
                    self.ctx.repeat(self.prefix.shape[0],1,1),     # (n_cls, n_ctx, dim)
                    self.suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )
        return torch.cat(
                [
                    self.prefix,  # (n_cls, 1, dim)
                    self.ctx,     # (n_cls, n_ctx, dim)
                    self.suffix,  # (n_cls, *, dim)
                ],
                dim=1,
        )

        
    def initialize_inference(self, text_encoder):
        n_prompts = self.get_text_prompts()
        text_feats = text_encoder(n_prompts, self.tokenized_prompts)
        self.text_feats = text_feats 
            
    def forward(self, vis_feat, text_encoder, train=True, local=False):
        n_prompts = self.get_text_prompts()
        if train:
            text_feats = text_encoder(n_prompts, self.tokenized_prompts)
            self.text_feats = text_feats 
        else:
            text_feats = self.text_feats
            
        vis_feats = vis_feat.float()#.to(self.device)
        tmp_res = self.implicit_mlp(vis_feats)
        
        vis_feats = tmp_res[...,:1024]  + vis_feats[...,:1024]

        self.vis_feats = vis_feats 
        logits = 2 * self.dropout(self.vis_feats) @ text_feats.T
        
    
        res = logits.float().reshape(*logits.shape[:-1], 2, -1).transpose(-2,-1).softmax(-1)[...,0]
             
        if local:
            return res
        else:
            return res.mean((-3,-2))
