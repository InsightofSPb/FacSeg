import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from torchvision import models as tvm

# ---------- CLIP head ----------
class CLIPHead(nn.Module):
    def __init__(self, model_name="ViT-B-32-quickgelu", pretrained="openai",
                 n_classes=8, device="cpu",
                 head_type="mlp", head_hidden=2048, head_dropout=0.1):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.visual = self.model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.proj_dim = self.visual.output_dim
        if head_type == "linear":
            self.head = nn.Linear(self.proj_dim, n_classes, bias=True)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.proj_dim, head_hidden, bias=True),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        with torch.no_grad():
            feats = self.visual(x)
            feats = F.normalize(feats.float(), dim=-1)
        logits = self.head(feats)
        return logits, feats

@torch.no_grad()
def init_head_from_text(head_module, class_names, model_name="ViT-B-32-quickgelu", pretrained="openai", device="cpu"):
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    templates = [
        "a building facade with {}",
        "{} on a wall",
        "structural damage: {} on masonry",
        "a photo of {} on a facade",
    ]
    W = []
    for name in class_names:
        phrases = [t.format(name.replace("_", " ").lower()) for t in templates]
        txt = tokenizer(phrases).to(device)
        emb = model.encode_text(txt).float()
        emb = F.normalize(emb, dim=-1).mean(0, keepdim=True)
        W.append(emb)
    W = torch.cat(W, dim=0)
    if isinstance(head_module, nn.Linear):
        head_module.weight.data.copy_(W)
        nn.init.zeros_(head_module.bias)
    else:
        for m in head_module.modules():
            if isinstance(m, nn.Linear) and m.out_features == W.shape[0]:
                m.weight.data.copy_(W)
                nn.init.zeros_(m.bias)
                break

# ---------- OVSeg (CLIP-ViT + light decoder) ----------
def _clip_spatial_tokens(visual, x):
    """
    Достаём токены из CLIP-ViT и корректно добавляем positional_embedding,
    интерполируя grid-часть под текущий L (= (H/ps)*(W/ps)).
    Работает для любых квадратных входов (например, 224, 336, 448, 512 ...).
    """
    B, _, H, W = x.shape

    # patchify -> (B, L, C)
    x = visual.conv1(x)
    x = x.reshape(B, visual.conv1.out_channels, -1).permute(0, 2, 1)  # B,L,C
    L = x.shape[1]  # число spatial-токенов

    # class token
    cls = visual.class_embedding.to(x.dtype).unsqueeze(0).expand(B, -1, -1)  # B,1,C

    # positional embedding [1+L0, C] -> интерполируем до [1+L, C]
    pos = visual.positional_embedding.to(x.dtype)  # (1+L0, C) или (1,1+L0,C)
    if pos.ndim == 2:
        pos = pos.unsqueeze(0)  # 1, 1+L0, C
    pos_cls, pos_grid = pos[:, :1, :], pos[:, 1:, :]  # 1,1,C  и  1,L0,C

    L0 = pos_grid.shape[1]
    if L != L0:
        gs0 = int(L0 ** 0.5)     # исходная решётка (обычно 14)
        gs  = int(L  ** 0.5)     # новая решётка (например 32 при 512x512)
        # 1,L0,C -> 1,C,gs0,gs0
        pos_grid = pos_grid.reshape(1, gs0, gs0, -1).permute(0, 3, 1, 2)
        pos_grid = torch.nn.functional.interpolate(pos_grid, size=(gs, gs), mode="bicubic", align_corners=False)
        # обратно: 1,C,gs,gs -> 1,L,C
        pos_grid = pos_grid.permute(0, 2, 3, 1).reshape(1, gs * gs, -1)

    pos = torch.cat([pos_cls, pos_grid], dim=1)     # 1, 1+L, C
    x = torch.cat([cls, x], dim=1) + pos            # B, 1+L, C

    # transformer
    x = visual.ln_pre(x) if hasattr(visual, "ln_pre") else x
    x = x.permute(1, 0, 2)
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    x = visual.ln_post(x) if hasattr(visual, "ln_post") else x

    # обратно в 2D map
    L = x.shape[1] - 1
    gs = int(L ** 0.5)
    x = x[:, 1:, :].permute(0, 2, 1).reshape(B, x.shape[-1], gs, gs)
    return x

class OvSegDecoder(nn.Module):
    def __init__(self, in_dim, out_classes, decoder_channels=256, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, decoder_channels, 1)
        self.block = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity(),
            nn.Conv2d(decoder_channels, decoder_channels, 3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(decoder_channels, out_classes, 1)

    def forward(self, feat_map, out_hw):
        x = self.proj(feat_map)
        x = torch.nn.functional.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        x = self.block(x)
        logits = self.head(x)
        return logits

class OvSegModel(nn.Module):
    """
    CLIP-ViT backbone (open_clip) + лёгкий декодер -> dict(out=логиты).
    Выход: C+1 каналов (фон+классы)
    """
    def __init__(self, model_name, pretrained, n_classes, freeze_backbone=False,
                 decoder_channels=256, decoder_dropout=0.0, device="cpu"):
        super().__init__()
        self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
        self.visual = self.clip_model.visual
        if freeze_backbone:
            for p in self.visual.parameters():
                p.requires_grad = False
        in_dim = getattr(self.visual, "width", None)
        if in_dim is None:
            in_dim = int(self.visual.ln_post.normalized_shape[0])
        self.decoder = OvSegDecoder(in_dim, n_classes + 1, decoder_channels, dropout=decoder_dropout)

    def forward(self, x):
        feat = _clip_spatial_tokens(self.visual, x)
        logits = self.decoder(feat, out_hw=(x.shape[-2], x.shape[-1]))
        return {"out": logits}

# ---------- torchvision baseline ----------
def get_seg_model(name: str, num_classes: int, weights_tag="imagenet"):
    name = name.lower()
    if name in ["deeplab", "deeplabv3_resnet50", "deeplabv3"]:
        m = tvm.segmentation.deeplabv3_resnet50(weights="DEFAULT" if weights_tag else None)
        m.classifier[4] = nn.Conv2d(256, num_classes + 1, 1)
        return m
    elif name in ["fcn", "fcn_resnet50", "fcn50"]:
        m = tvm.segmentation.fcn_resnet50(weights="DEFAULT" if weights_tag else None)
        m.classifier[4] = nn.Conv2d(512, num_classes + 1, 1)
        return m
    else:
        raise ValueError(f"Unknown seg model: {name}")
