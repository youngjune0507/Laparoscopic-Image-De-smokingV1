#source : https://github.com/IDKiro/DehazeFormer
# Install required packages
!pip install torch torchvision tqdm opencv-python numpy matplotlib scikit-image timm

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from timm.models.layers import to_2tuple, trunc_normal_
import math
from torch.nn.init import _calculate_fan_in_and_fan_out

# Robust Layer Normalization
class RLN(nn.Module):
    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad
        self.gamma = nn.Parameter(torch.ones((1, dim, 1, 1)))  # Renamed weight to gamma
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)))  # Renamed bias to beta
        self.meta_conv1 = nn.Conv2d(1, dim, 1)
        self.meta_conv2 = nn.Conv2d(1, dim, 1)
        trunc_normal_(self.meta_conv1.weight, std=.02)
        nn.init.constant_(self.meta_conv1.bias, 1)
        trunc_normal_(self.meta_conv2.weight, std=.02)
        nn.init.constant_(self.meta_conv2.bias, 0)

    def forward(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((x - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)
        norm_x = (x - mean) / std
        if self.detach_grad:
            scale, offset = self.meta_conv1(std.detach()), self.meta_conv2(mean.detach())
        else:
            scale, offset = self.meta_conv1(std), self.meta_conv2(mean)
        out = norm_x * self.gamma + self.beta
        return out, scale, offset

# MLP Block
class Mlp(nn.Module):
    def __init__(self, total_depth, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim
        self.total_depth = total_depth
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_dim, out_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            gain = (8 * self.total_depth) ** (-1/4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(layer.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(layer.weight, std=std)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.net(x)

# Split image into windows
def window_split(x, win_size):
    B, H, W, C = x.shape
    x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size**2, C)
    return windows

# Merge windows back to image
def window_merge(windows, win_size, H, W):
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# Compute relative positions for attention
def get_rel_pos(win_size):
    coords_h = torch.arange(win_size)
    coords_w = torch.arange(win_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
    coords_flat = torch.flatten(coords, 1)
    rel_pos = coords_flat[:, :, None] - coords_flat[:, None, :]
    rel_pos = rel_pos.permute(1, 2, 0).contiguous()
    rel_pos_log = torch.sign(rel_pos) * torch.log(1. + rel_pos.abs())
    return rel_pos_log

# Window-based Attention
class WinAttention(nn.Module):
    def __init__(self, dim, win_size, num_heads):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        rel_pos = get_rel_pos(self.win_size)
        self.register_buffer("rel_pos", rel_pos)
        self.meta_net = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv):
        B_, N, _ = qkv.shape
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        rel_pos_bias = self.meta_net(self.rel_pos)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1).contiguous()
        attn = attn + rel_pos_bias.unsqueeze(0)
        attn = self.softmax(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        return out

# Attention Module
class Attention(nn.Module):
    def __init__(self, total_depth, dim, num_heads, win_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.total_depth = total_depth
        self.use_attn = use_attn
        self.conv_type = conv_type
        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )
        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
        if self.conv_type == 'DWConv' or use_attn:
            self.value_conv = nn.Conv2d(dim, dim, 1)  # Renamed V to value_conv
            self.out_conv = nn.Conv2d(dim, dim, 1)   # Renamed proj to out_conv
        if use_attn:
            self.qk_conv = nn.Conv2d(dim, dim * 2, 1)  # Renamed QK to qk_conv
            self.attn = WinAttention(dim, win_size, num_heads)
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, nn.Conv2d):
            w_shape = layer.weight.shape
            if w_shape[0] == self.dim * 2:  # QK weights
                fan_in, fan_out = _calculate_fan_in_and_fan_out(layer.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(layer.weight, std=std)
            else:
                gain = (8 * self.total_depth) ** (-1/4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(layer.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(layer.weight, std=std)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)

    def pad_image(self, x, shift=False):
        _, _, h, w = x.size()
        pad_h = (self.win_size - h % self.win_size) % self.win_size
        pad_w = (self.win_size - w % self.win_size) % self.win_size
        if shift:
            x = F.pad(x, (self.shift_size, (self.win_size-self.shift_size+pad_w) % self.win_size,
                          self.shift_size, (self.win_size-self.shift_size+pad_h) % self.win_size), mode='reflect')
        else:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        if self.conv_type == 'DWConv' or self.use_attn:
            v = self.value_conv(x)
        if self.use_attn:
            qk = self.qk_conv(x)
            qkv = torch.cat([qk, v], dim=1)
            shifted_qkv = self.pad_image(qkv, self.shift_size > 0)
            Ht, Wt = shifted_qkv.shape[2:]
            shifted_qkv = shifted_qkv.permute(0, 2, 3, 1)
            qkv_win = window_split(shifted_qkv, self.win_size)
            attn_win = self.attn(qkv_win)
            shifted_out = window_merge(attn_win, self.win_size, Ht, Wt)
            out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
            attn_out = out.permute(0, 3, 1, 2)
            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(v)
                out = self.out_conv(conv_out + attn_out)
            else:
                out = self.out_conv(attn_out)
        else:
            if self.conv_type == 'Conv':
                out = self.conv(x)
            elif self.conv_type == 'DWConv':
                out = self.out_conv(self.conv(v))
        return out

# Transformer Block
class TransBlock(nn.Module):
    def __init__(self, total_depth, dim, num_heads, mlp_ratio=4., norm_layer=nn.LayerNorm,
                 mlp_norm=False, win_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm
        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(total_depth, dim, num_heads=num_heads, win_size=win_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)
        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.mlp = Mlp(total_depth, dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x):
        residual = x
        if self.use_attn:
            x, scale, offset = self.norm1(x)
        x = self.attn(x)
        if self.use_attn:
            x = x * scale + offset
        x = residual + x
        residual = x
        if self.use_attn and self.mlp_norm:
            x, scale, offset = self.norm2(x)
        x = self.mlp(x)
        if self.use_attn and self.mlp_norm:
            x = x * scale + offset
        x = residual + x
        return x

# Basic Transformer Layer
class TransformerLayer(nn.Module):
    def __init__(self, total_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, win_size=8, attn_ratio=0., attn_loc='last', conv_type=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        attn_depth = attn_ratio * depth
        if attn_loc == 'last':
            use_attns = [i >= depth-attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransBlock(total_depth=total_depth, dim=dim, num_heads=num_heads,
                       mlp_ratio=mlp_ratio, norm_layer=norm_layer, win_size=win_size,
                       shift_size=0 if (i % 2 == 0) else win_size // 2,
                       use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

# Patch Embedding
class PatchEncoder(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        kernel_size = kernel_size or patch_size
        self.conv = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        return self.conv(x)

# Patch Unembedding
class PatchDecoder(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        kernel_size = kernel_size or 1
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        return self.net(x)

# Selective Kernel Fusion
class SKFuse(nn.Module):
    def __init__(self, dim, num_inputs=2, reduction=8):
        super().__init__()
        self.num_inputs = num_inputs
        reduced_dim = max(int(dim/reduction), 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, reduced_dim, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, dim*num_inputs, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        B, C, H, W = inputs[0].shape
        feats = torch.cat(inputs, dim=1).view(B, self.num_inputs, C, H, W)
        feats_sum = torch.sum(feats, dim=1)
        attn = self.mlp(self.pool(feats_sum))
        attn = self.softmax(attn.view(B, self.num_inputs, C, 1, 1))
        out = torch.sum(feats*attn, dim=1)
        return out

# DehazeFormer Model
class DehazeFormer(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, win_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super().__init__()
        self.patch_size = 4
        self.win_size = win_size
        self.mlp_ratios = mlp_ratios
        self.encoder1 = PatchEncoder(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
        self.stage1 = TransformerLayer(sum(depths), embed_dims[0], depths[0], num_heads[0],
                                      mlp_ratios[0], norm_layer[0], win_size, attn_ratio[0], 'last', conv_type[0])
        self.down1 = PatchEncoder(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.skip1 = nn.Conv2d(embed_dims[0], embed_dims[0], 1)
        self.stage2 = TransformerLayer(sum(depths), embed_dims[1], depths[1], num_heads[1],
                                      mlp_ratios[1], norm_layer[1], win_size, attn_ratio[1], 'last', conv_type[1])
        self.down2 = PatchEncoder(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.skip2 = nn.Conv2d(embed_dims[1], embed_dims[1], 1)
        self.stage3 = TransformerLayer(sum(depths), embed_dims[2], depths[2], num_heads[2],
                                      mlp_ratios[2], norm_layer[2], win_size, attn_ratio[2], 'last', conv_type[2])
        self.up1 = PatchDecoder(patch_size=2, out_chans=embed_dims[3], embed_dim=embed_dims[2])
        assert embed_dims[1] == embed_dims[3]
        self.fuse1 = SKFuse(embed_dims[3])
        self.stage4 = TransformerLayer(sum(depths), embed_dims[3], depths[3], num_heads[3],
                                      mlp_ratios[3], norm_layer[3], win_size, attn_ratio[3], 'last', conv_type[3])
        self.up2 = PatchDecoder(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
        assert embed_dims[0] == embed_dims[4]
        self.fuse2 = SKFuse(embed_dims[4])
        self.stage5 = TransformerLayer(sum(depths), embed_dims[4], depths[4], num_heads[4],
                                      mlp_ratios[4], norm_layer[4], win_size, attn_ratio[4], 'last', conv_type[4])
        self.decoder = PatchDecoder(patch_size=1, out_chans=out_chans, embed_dim=embed_dims[4], kernel_size=3)

    def pad_input(self, x):
        _, _, h, w = x.size()
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.encoder1(x)
        x = self.stage1(x)
        skip1 = x
        x = self.down1(x)
        x = self.stage2(x)
        skip2 = x
        x = self.down2(x)
        x = self.stage3(x)
        x = self.up1(x)
        x = self.fuse1([x, self.skip2(skip2)]) + x
        x = self.stage4(x)
        x = self.up2(x)
        x = self.fuse2([x, self.skip1(skip1)]) + x
        x = self.stage5(x)
        x = self.decoder(x)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.pad_input(x)
        feat = self.forward_features(x)
        K, B = torch.split(feat, (1, 3), dim=1)
        out = K * x - B + x
        out = out[:, :, :H, :W]
        return out

# DehazeFormer variants
def dehazeformer_tiny():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[4, 4, 4, 2, 2],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[0, 1/2, 1, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])

def dehazeformer_small():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[8, 8, 8, 4, 4],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])

def dehazeformer_base():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])

def dehazeformer_deep():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[32, 32, 32, 16, 16],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])

def dehazeformer_wide():
    return DehazeFormer(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 8, 8],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])

def dehazeformer_medium():
    return DehazeFormer(
        embed_dims=[24, 48, 96, 48, 24],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[12, 12, 12, 6, 6],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])

def dehazeformer_large():
    return DehazeFormer(
        embed_dims=[48, 96, 192, 96, 48],
        mlp_ratios=[2., 4., 4., 2., 2.],
        depths=[16, 16, 16, 12, 12],
        num_heads=[2, 4, 6, 1, 1],
        attn_ratio=[1/4, 1/2, 3/4, 0, 0],
        conv_type=['Conv', 'Conv', 'Conv', 'Conv', 'Conv'])

# Dataset for Dehazing
class HazeDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        self.images = sorted([f for f in os.listdir(root_dir) if not f.endswith('_gt.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.root_dir, self.images[idx])
        clean_path = os.path.join(self.root_dir, self.images[idx].replace('.png', '_gt.png'))
        input_img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB) / 255.0
        
        if self.is_train:
            input_img = cv2.resize(input_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            clean_img = cv2.resize(clean_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = input_img.shape[:2]
            h = h - (h % 4)
            w = w - (w % 4)
            input_img = cv2.resize(input_img, (w, h), interpolation=cv2.INTER_LINEAR)
            clean_img = cv2.resize(clean_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float()
        clean_tensor = torch.from_numpy(clean_img.transpose(2, 0, 1)).float()
        return input_tensor, clean_tensor

# Training and Evaluation Function
def train_and_evaulate(data_path, epochs=15, batch_size=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    net = 'size of the model'().to(device)  # Use small model for efficiency
    loss_fn = nn.L1Loss()  # L1 loss for training
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)

    train_data = HazeDataset(data_path, is_train=True)
    val_data = HazeDataset(data_path, is_train=False)
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, _ = random_split(train_data, [train_size, val_size])
    _, val_subset = random_split(val_data, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=0)

    for epoch in range(epochs):
        net.train()
        total_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for input_img, clean_img in progress:
            input_img, clean_img = input_img.to(device), clean_img.to(device)
            opt.zero_grad()
            output_img = net(input_img)
            loss = loss_fn(output_img, clean_img)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            progress.set_postfix({'Loss': total_loss / (progress.n + 1)})
        avg_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss:.4f}")

    print("Evaluating on validation set...")
    net.eval()
    psnr_scores = []
    ssim_scores = []
    with torch.no_grad():
        for input_img, clean_img in tqdm(val_loader, desc="Evaluation"):
            input_img, clean_img = input_img.to(device), clean_img.to(device)
            output_img = net(input_img)
            output_img = F.interpolate(output_img, size=(clean_img.shape[2], clean_img.shape[3]),
                                      mode='bilinear', align_corners=False)
            output_np = output_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            clean_np = clean_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr_scores.append(psnr(clean_np, output_np, data_range=1.0))
            ssim_scores.append(ssim(clean_np, output_np, multichannel=True, data_range=1.0,
                                    win_size=11, channel_axis=2))

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    print(f"Final PSNR: {avg_psnr:.4f}")
    print(f"Final SSIM: {avg_ssim:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(psnr_scores) + 1), psnr_scores, marker='o', label='PSNR')
    plt.axhline(y=avg_psnr, color='r', linestyle='--', label=f'Avg: {avg_psnr:.2f}')
    plt.xlabel('Image')
    plt.ylabel('PSNR')
    plt.title('PSNR Scores')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(ssim_scores) + 1), ssim_scores, marker='o', label='SSIM')
    plt.axhline(y=avg_ssim, color='r', linestyle='--', label=f'Avg: {avg_ssim:.2f}')
    plt.xlabel('Image')
    plt.ylabel('SSIM')
    plt.title('SSIM Scores')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    torch.save(net.state_dict(), 'dehazeformer_model.pth')
    print("Saved model as 'dehazeformer_model.pth'")

# Run everything
if __name__ == "__main__":
    dataset_path = 'path to the dataset'
    train_and_evaluate(dataset_path, epochs=15, batch_size=4)
