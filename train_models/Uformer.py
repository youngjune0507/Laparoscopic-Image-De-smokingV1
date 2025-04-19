#source : https://github.com/ZhendongWang6/Uformer/tree/main, https://github.com/lucidrains/uformer-pytorch/tree/main




import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from math import log, pi, sqrt
from functools import partial
from einops import rearrange, repeat

# Uformer
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, depth=1):
    return val if isinstance(val, tuple) else (val,) * depth

def apply_rotary_emb(q, k, pos_emb):
    sin, cos = pos_emb
    dim_rotary = sin.shape[-1]
    (q, q_pass), (k, k_pass) = map(lambda t: (t[..., :dim_rotary], t[..., dim_rotary:]), (q, k))
    q, k = map(lambda t: (t * cos) + (rotate_every_two(t) * sin), (q, k))
    q, k = map(lambda t: torch.cat(t, dim=-1), ((q, q_pass), (k, k_pass)))
    return q, k

def rotate_every_two(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d j -> ... (d j)')

class AxialRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10):
        super().__init__()
        self.dim = dim
        scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        self.register_buffer('scales', scales)

    def forward(self, x):
        device, dtype, h, w = x.device, x.dtype, *x.shape[-2:]
        seq_x = torch.linspace(-1., 1., steps=h, device=device)
        seq_x = seq_x.unsqueeze(-1)
        seq_y = torch.linspace(-1., 1., steps=w, device=device)
        seq_y = seq_y.unsqueeze(-1)
        scales = self.scales[(*((None,) * (len(seq_x.shape) - 1)), ...)]
        scales = scales.to(x)
        seq_x = seq_x * scales * pi
        seq_y = seq_y * scales * pi
        x_sinu = repeat(seq_x, 'i d -> i j d', j=w)
        y_sinu = repeat(seq_y, 'j d -> i j d', i=h)
        sin = torch.cat((x_sinu.sin(), y_sinu.sin()), dim=-1)
        cos = torch.cat((x_sinu.cos(), y_sinu.cos()), dim=-1)
        sin, cos = map(lambda t: rearrange(t, 'i j d -> i j d'), (sin, cos))
        sin, cos = map(lambda t: repeat(t, 'i j d -> () i j (d r)', r=2), (sin, cos))
        return sin, cos

class TimeSinuPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = einsum('i, j -> i j', x, emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, window_size=16):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.window_size = window_size
        inner_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, inner_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x, skip=None, time_emb=None, pos_emb=None):
        h, w, b = self.heads, self.window_size, x.shape[0]
        if exists(time_emb):
            time_emb = rearrange(time_emb, 'b c -> b c () ()')
            x = x + time_emb
        q = self.to_q(x)
        kv_input = x
        if exists(skip):
            kv_input = torch.cat((kv_input, skip), dim=0)
        k, v = self.to_kv(kv_input).chunk(2, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) x y c', h=h), (q, k, v))
        if exists(pos_emb):
            q, k = apply_rotary_emb(q, k, pos_emb)
        q, k, v = map(lambda t: rearrange(t, 'b (x w1) (y w2) c -> (b x y) (w1 w2) c', w1=w, w2=w), (q, k, v))
        if exists(skip):
            k, v = map(lambda t: rearrange(t, '(r b) n d -> b (r n) d', r=2), (k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h x y) (w1 w2) c -> b (h c) (x w1) (y w2)', b=b, h=h, y=x.shape[-1] // w, w1=w, w2=w)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        hidden_dim = dim * mult
        self.project_in = nn.Conv2d(dim, hidden_dim, 1)
        self.project_out = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x, time_emb=None):
        x = self.project_in(x)
        if exists(time_emb):
            time_emb = rearrange(time_emb, 'b c -> b c () ()')
            x = x + time_emb
        return self.project_out(x)

class Block(nn.Module):
    def __init__(self, dim, depth, dim_head=64, heads=8, ff_mult=4, window_size=16, time_emb_dim=None, rotary_emb=True):
        super().__init__()
        self.attn_time_emb = None
        self.ff_time_emb = None
        if exists(time_emb_dim):
            self.attn_time_emb = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            self.ff_time_emb = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim * ff_mult))
        self.pos_emb = AxialRotaryEmbedding(dim_head) if rotary_emb else None
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, dim_head=dim_head, heads=heads, window_size=window_size)),
                PreNorm(dim, FeedForward(dim, mult=ff_mult))
            ]))

    def forward(self, x, skip=None, time=None):
        attn_time_emb = None
        ff_time_emb = None
        if exists(time):
            assert exists(self.attn_time_emb) and exists(self.ff_time_emb), 'time_emb_dim must be given'
            attn_time_emb = self.attn_time_emb(time)
            ff_time_emb = self.ff_time_emb(time)
        pos_emb = None
        if exists(self.pos_emb):
            pos_emb = self.pos_emb(x)
        for attn, ff in self.layers:
            x = attn(x, skip=skip, time_emb=attn_time_emb, pos_emb=pos_emb) + x
            x = ff(x, time_emb=ff_time_emb) + x
        return x

class Uformer(nn.Module):
    def __init__(self, dim=64, channels=3, stages=4, num_blocks=2, dim_head=64, window_size=16, heads=8, ff_mult=4, time_emb=False, input_channels=None, output_channels=None):
        super().__init__()
        input_channels = default(input_channels, channels)
        output_channels = default(output_channels, channels)
        self.to_time_emb = None
        time_emb_dim = None
        if time_emb:
            time_emb_dim = dim
            self.to_time_emb = nn.Sequential(
                TimeSinuPosEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            )
        self.project_in = nn.Sequential(
            nn.Conv2d(input_channels, dim, 3, padding=1),
            nn.GELU()
        )
        self.project_out = nn.Sequential(
            nn.Conv2d(dim, output_channels, 3, padding=1),
        )
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        heads, window_size, dim_head, num_blocks = map(partial(cast_tuple, depth=stages), (heads, window_size, dim_head, num_blocks))
        for ind, heads, window_size, dim_head, num_blocks in zip(range(stages), heads, window_size, dim_head, num_blocks):
            is_last = ind == (stages - 1)
            self.downs.append(nn.ModuleList([
                Block(dim, depth=num_blocks, dim_head=dim_head, heads=heads, ff_mult=ff_mult, window_size=window_size, time_emb_dim=time_emb_dim),
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1)
            ]))
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(dim * 2, dim, 2, stride=2),
                Block(dim, depth=num_blocks, dim_head=dim_head, heads=heads, ff_mult=ff_mult, window_size=window_size, time_emb_dim=time_emb_dim)
            ]))
            dim *= 2
            if is_last:
                self.mid = Block(dim=dim, depth=num_blocks, dim_head=dim_head, heads=heads, ff_mult=ff_mult, window_size=window_size, time_emb_dim=time_emb_dim)

    def forward(self, x, time=None):
        if exists(time):
            assert exists(self.to_time_emb), 'time_emb must be set to true'
            time = time.to(x)
            time = self.to_time_emb(time)
        x = self.project_in(x)
        skips = []
        for block, downsample in self.downs:
            x = block(x, time=time)
            skips.append(x)
            x = downsample(x)
        x = self.mid(x, time=time)
        for (upsample, block), skip in zip(reversed(self.ups), reversed(skips)):
            x = upsample(x)
            x = block(x, skip=skip, time=time)
        x = self.project_out(x)
        return x

# Losses
class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

# dataset
class DehazeDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.hazy_images = sorted([f for f in os.listdir(data_dir) if not f.endswith('_gt.png')])

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.data_dir, self.hazy_images[idx])
        gt_path = os.path.join(self.data_dir, self.hazy_images[idx].replace('.png', '_gt.png'))
        hazy_img = cv2.imread(hazy_path, cv2.IMREAD_COLOR)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB) / 255.0
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) / 255.0
        hazy_img = torch.from_numpy(hazy_img.transpose(2, 0, 1)).float()
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).float()
        return hazy_img, gt_img, hazy_path

# Training and Evaluating
def train_and_evaluate(data_dir, num_epochs=10, batch_size=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = Uformer(dim=64, channels=3, stages=4, num_blocks=2, dim_head=64, window_size=16, heads=8, ff_mult=4, time_emb=False).to(device)
    char_loss = CharbonnierLoss(eps=1e-3)
    tv_loss = TVLoss(tv_loss_weight=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_dataset = DehazeDataset(data_dir, train=True)
    val_dataset = DehazeDataset(data_dir, train=False)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    _, val_dataset = random_split(val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for hazy_img, gt_img, _ in train_loader:
                hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
                optimizer.zero_grad()
                dehaze_img = model(hazy_img)
                c_loss = char_loss(dehaze_img, gt_img)
                t_loss = tv_loss(dehaze_img)
                loss = c_loss + t_loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'Loss': train_loss / (pbar.n + 1)})
                pbar.update(1)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

    print("Generating comparison images...")
    model.eval()
    os.makedirs('comparisons', exist_ok=True)
    with torch.no_grad():
        for idx, (hazy_img, gt_img, hazy_path) in enumerate(tqdm(val_loader, desc="Generating Comparisons", unit="image")):
            hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
            dehaze_img = model(hazy_img)
            hazy_np = hazy_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            dehaze_np = dehaze_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = gt_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(hazy_np)
            plt.title('Hazy Image')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(dehaze_np)
            plt.title('Dehazed Image')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(gt_np)
            plt.title('Ground Truth')
            plt.axis('off')
            plt.tight_layout()
            img_name = os.path.basename(hazy_path[0]).replace('.png', '')
            save_path = f'comparisons/comparison_{img_name}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()
            print(f"Saved comparison image: {save_path}")

    torch.save(model.state_dict(), 'uformer_trained.pth')
    print("Model saved as 'uformer_trained.pth'")

# Execution
if __name__ == "__main__":
    data_dir = 'path to your dataset'
    train_and_evaluate(data_dir, num_epochs=10, batch_size=2)
