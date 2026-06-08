"""
PRISM — Video Deepfake Detector (Speed-optimised build)
Architecture: EfficientNet-B0 (1280-dim) → Linear(256) → SinusoidalPE
              → TransformerEncoder (2 layers, 8 heads) → CLS token → MLP → real/fake

Chosen over B4 for RTX 3050 6GB time-constrained training:
  - B0 backbone: 5.3M params vs B4's 19M  →  ~3× faster per batch
  - Same Transformer head as original design
  - Fits comfortably at batch_size=32, seq_len=4
"""

import math
import torch
import torch.nn as nn
import timm


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                        * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.norm2(x + self.ff(x))
        return x


class DeepfakeDetector(nn.Module):
    """
    Input:  (B, T, 3, 224, 224)
    Output: (B,)  logit — positive = fake
    """
    def __init__(self, d_model=256, n_heads=8, n_layers=2,
                 dropout=0.1, pretrained=True):
        super().__init__()

        # EfficientNet-B0: 1280-dim output, much lighter than B4
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=pretrained,
            num_classes=0, global_pool="avg"
        )
        backbone_dim = self.backbone.num_features  # 1280

        self.proj      = nn.Sequential(
            nn.Linear(backbone_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_enc   = SinusoidalPE(d_model, dropout=dropout)
        self.transformer = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats  = self.backbone(x.view(B * T, C, H, W))   # (B*T, 1280)
        feats  = self.proj(feats).view(B, T, -1)          # (B, T, d)
        cls    = self.cls_token.expand(B, -1, -1)
        tokens = self.pos_enc(torch.cat([cls, feats], 1)) # (B, T+1, d)
        out    = self.transformer(tokens)
        return self.head(out[:, 0]).squeeze(-1)            # (B,)


def build_model(pretrained=True, d_model=256, n_heads=8, n_layers=2,
                dropout=0.1, checkpoint_path=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                             dropout=dropout, pretrained=pretrained).to(device)
    if checkpoint_path:
        ckpt  = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"[build_model] Loaded {checkpoint_path}")
    return model