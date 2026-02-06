# REViT (Recurrent Event-rgb Vision Transoformer) policy 
# policy model

import torch 
import  torch.nn as nn
from torchvision import models

"""
Contains Class : PatchEmbedding, PositionalEncoder, SelfAttentionBlock, MLPBlock, TransformerEncoderBlock, LSTMBlock, PolicyHeadArm, PolicyHeadNav.

"""

class PatchEmbedding(nn.Module):
    """
    Fuse event + rgb frames 
    """
    def __init__(self, in_channels:int=5, # 3 rgb + 2 event
                        emd_dim:int=64,
                        patch_size:int=16):
        
        super().__init__()

        self.proj = nn.Conv2d(
                    in_channels, emd_dim,
                    kernel_size=patch_size, stride=patch_size
        )



    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)  # (B, N, d_model)
        return x

class MobileNetV3SmallEncoder(nn.Module):
    """
    MobileNetV3-Small backbone -> global pooled embedding.
    - Works for RGB (in_channels=3) and event frames (in_channels=1/2/B).
    - If pretrained=True and in_channels!=3, first conv is adapted from ImageNet weights.
    """
    def __init__(self,
                 in_channels: int = 3,
                 embed_dim: int = 256,
                 pretrained: bool = True,
                 freeze_bn: bool = False):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=weights)

        # --- Patch first conv if input channels differ  (Events)---  
        # First conv lives at m.features[0][0] (Conv2d)
        first_conv = m.features[0][0]
        if in_channels != first_conv.in_channels:
            new_first = nn.Conv2d(
                in_channels, first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False
            )
            if pretrained:
                with torch.no_grad():
                    w = first_conv.weight  # [out, 3, k, k]
                    base = w.mean(dim=1, keepdim=True)  # [out,1,k,k]
                    new_first.weight.copy_(base.repeat(1, in_channels, 1, 1))
            m.features[0][0] = new_first

        self.features = m.features  # feature extractor trunk
        self.last_dim = m.classifier[0].in_features  # 576 for v3-small
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(self.last_dim, embed_dim)

        if freeze_bn:
            self._freeze_batchnorm()

    def _freeze_batchnorm(self):
        # Keep BN running stats but don’t update affine params
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, x):
        """
        x: (B, C, H, W) — e.g., 224x224
        returns: (B, embed_dim)
        """
        x = self.features(x)       # (B, last_dim, H/32, W/32) typically
        x = self.pool(x).flatten(1)  # (B, last_dim)
        z = self.proj(x)           # (B, embed_dim)
        return z



class PostionalEncoder(nn.Module):
    """
    Orders the token positions
    """
    def __init__(self, num_tokens:int=1601,
                        embed_dim:int=64):
        super().__init__()

        self.pos = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)


    def forward(self, x):
          
        return x + self.pos

class SelfAttentionBlock(nn.Module):
    """
    Multi‐head self‐attention block with pre‐LayerNorm.
    """
    def __init__(self,embedding_dim:int = 128,
                      num_heads:int = 4,
                      attn_dropout:float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.multihead_attn = nn.MultiheadAttention(
                                    embed_dim=embedding_dim,
                                    num_heads=num_heads,
                                    dropout=attn_dropout,
                                    batch_first=True
        )

    def forward(self, x):
        # x: (B, S, embedding_dim)
        x_norm = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            need_weights=False
        )
        return attn_output


class MLPBlock(nn.Module):
    """
    Two‐layer MLP with pre‐LayerNorm and GELU activation.
    """
    def __init__(self, embedding_dim:int = 128,
                        mlp_size:int = 256,
                        dropout:float = 0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
                                nn.Linear(in_features=embedding_dim, out_features=mlp_size),
                                nn.GELU(),
                                nn.Dropout(p=dropout),
                                nn.Linear(in_features=mlp_size, out_features=embedding_dim),
                                nn.Dropout(p=dropout)
        )

    def forward(self, x):
        # x: (B, S, embedding_dim)
        return self.mlp(self.layer_norm(x))
    

class TransformerEncoderBlock(nn.Module):
    """
    Single transformer encoder block:  
      x → x + MSA(x) → x + MLP(x)
    with pre‐LayerNorm in each sub‐block.
    """
    def __init__(self,embedding_dim: int = 128,
                    num_heads: int = 4,
                    mlp_size: int = 256,
                    mlp_dropout: float = 0.1,
                    attn_dropout: float = 0.1):
        super().__init__()

        self.msa_block = SelfAttentionBlock(
                                        embedding_dim=embedding_dim,
                                        num_heads=num_heads,
                                        attn_dropout=attn_dropout
        )

        self.mlp_block = MLPBlock(
                                embedding_dim=embedding_dim,
                                mlp_size=mlp_size,
                                dropout=mlp_dropout
        )

    def forward(self, x):
        # x: (B, S, embedding_dim)
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class LSTMBlock(nn.Module):
    """
    Single‐step LSTMCell wrapper for recurrent memory.
    """
    def __init__(self,
                 input_dim: int = 64,
                 hidden_dim: int = 32):
        super().__init__()

        self.cell = nn.LSTMCell(input_dim, hidden_dim)

    def init_hidden(self, batch_size: int, device: torch.device):
        """
        Create initial (h0, c0) zeros for a new episode.
        """
        h0 = torch.zeros(batch_size, self.cell.hidden_size, device=device)
        c0 = torch.zeros(batch_size, self.cell.hidden_size, device=device)
        return h0, c0

    def forward(self,
                x: torch.Tensor,
                hidden: tuple):
        """
        x: (B, input_dim)      — the CLS token embedding from your transformer
        hidden: (h, c)         — each (B, hidden_dim)
        returns: (h_next, c_next)
        """
        h, c = self.cell(x, hidden)
        return h, c
    
class PolicyHeadArm(nn.Module):
    """
    Takes in LSTM input and passes through MLP block and outputs the Action of 
    the robotic arm (Roll, Pitch, Yaw, x, y, z, Gripper)
    """
    def __init__(self, hidden_dim:int=32,
                         mlp_dim:int=32,
                        dropout:float=0.1):
        super().__init__()

        self.net = nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, mlp_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(mlp_dim, 7)  # exactly 7 outputs for robot arm
        )

    def forward(self, h):
        return self.net(h)    
    
    
class PolicyHeadNav(nn.Module):
    def __init__(self, hidden_dim:int=32,
                         mlp_dim:int=32,
                        dropout:float=0.1):
        super().__init__()

        self.net = nn.Sequential(
                            nn.LayerNorm(hidden_dim),
                            nn.Linear(hidden_dim, mlp_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(mlp_dim, 2)  # exactly 2 outputs for Mobile Robot navigation
        )

    def forward(self, h):
        h = self.net(h)
        h = torch.tanh(h)
        return h   
    


class ENPNav(nn.Module):
    """

    """

    def __init__(self,  # Patch 
                        in_channels:int=5, # 3 rgb + 2 event
                        rgb_channels:int=3,
                        event_channels:int=2,
                        emd_dim:int=128,
                        # patch_size:int=16,

                        # Positional Encoder
                        # num_tokens:int=1601,

                        # Transformer head + mlp block
                        num_heads: int = 4,
                        mlp_size: int = 256,
                        mlp_dropout: float = 0.1,
                        attn_dropout: float = 0.1,

                        # lstm block
                        # hidden_dim: int = 32,

                        # policy head
                        mlp_dim:int=32,
                        dropout:float=0.1):
        super().__init__()

        self.rgb_encoder = MobileNetV3SmallEncoder(in_channels=rgb_channels,
                                                    embed_dim=emd_dim,
                                                    pretrained=True,
                                                    freeze_bn= True)
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False  # hard-freeze RGB
        
        self.evt_encoder = MobileNetV3SmallEncoder(in_channels=event_channels, embed_dim=emd_dim,
            pretrained=True, freeze_bn=False
        )

        self.encoder1 = TransformerEncoderBlock(embedding_dim=emd_dim,
                                                num_heads   =num_heads,
                                                mlp_size    =mlp_size,
                                                mlp_dropout =mlp_dropout,
                                                attn_dropout=attn_dropout)
        
        self.pos = nn.Parameter(torch.zeros(1, 2, emd_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.norm = nn.LayerNorm(emd_dim)

        self.policyhead = PolicyHeadNav(hidden_dim=emd_dim,
                                        mlp_dim   =mlp_dim,
                                        dropout   =dropout)

    def forward(self, rgb_img=None, evt_img=None):

        tokens = []

        if rgb_img is not None:
            rgb_z = self.rgb_encoder(rgb_img)             # (B, E)
            tokens.append(rgb_z.unsqueeze(1))             # (B, 1, E)

        if evt_img is not None:
            evt_z = self.evt_encoder(evt_img)             # (B, E)
            tokens.append(evt_z.unsqueeze(1))             # (B, 1, E)

        if len(tokens) == 0:
            raise ValueError("At least one of rgb_img or evt_img must be provided.")

        x = torch.cat(tokens, dim=1)                      # (B, n_tokens, E), n_tokens ∈ {1,2}

        # add positional embeddings for the number of tokens we actually have
        x = x + self.pos[:, :x.size(1), :]                # (B, n_tokens, E)

        # --- self-attention over 1 or 2 tokens ---
        x = self.encoder1(x)                              # (B, n_tokens, E)

        # pool tokens (mean works well; could also use CLS-style if you add one)
        fused = x.mean(dim=1)                             # (B, E)
        fused = self.norm(fused)

        # Policy Head
        action = self.policyhead(fused)                   # (B, 2)
        return action



