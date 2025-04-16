# source : https://github.com/c-yn/DSANet

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

# Basic Convolution Block
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2 if not transpose else kernel_size // 2 - 1
        layers = []
        if transpose:
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        if filter:
            self.cubic_11 = cubic_attention(in_channel//2, group=1, kernel=11)
            self.cubic_7 = cubic_attention(in_channel//2, group=1, kernel=7)
            self.pool_att = SpecAtte(in_channel)
        self.filter = filter

    def forward(self, x):
        out = self.conv1(x)
        if self.filter:
            out = self.pool_att(out)
            out = torch.chunk(out, 2, dim=1)
            out_11 = self.cubic_11(out[0])
            out_7 = self.cubic_7(out[1])
            out = torch.cat((out_11, out_7), dim=1)
        out = self.conv2(out)
        return out + x

# Encoder Block
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res-1)]
        layers.append(ResBlock(out_channel, out_channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Decoder Block
class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel) for _ in range(num_res-1)]
        layers.append(ResBlock(channel, channel, filter=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Spatial Context Module
class SCM(nn.Module):
    def __init__(self, out_plane):
        super(SCM, self).__init__()
        self.main = nn.Sequential(
            BasicConv(3, out_plane//4, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane//4, out_plane//2, kernel_size=1, stride=1, relu=True),
            BasicConv(out_plane//2, out_plane//2, kernel_size=3, stride=1, relu=True),
            BasicConv(out_plane//2, out_plane, kernel_size=1, stride=1, relu=False),
            nn.InstanceNorm2d(out_plane, affine=True)
        )

    def forward(self, x):
        return self.main(x)

# Feature Aggregation Module
class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

# Cubic Attention Module
class cubic_attention(nn.Module):
    def __init__(self, dim, group, kernel):
        super().__init__()
        self.H_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel)
        self.W_spatial_att = spatial_strip_att(dim, group=group, kernel=kernel, H=False)
        self.gamma = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        out = self.H_spatial_att(x)
        out = self.W_spatial_att(out)
        return self.gamma * out + x * self.beta

# Spatial Strip Attention
class spatial_strip_att(nn.Module):
    def __init__(self, dim, kernel=5, group=2, H=True):
        super().__init__()
        self.k = kernel
        pad = kernel // 2
        self.kernel = (1, kernel) if H else (kernel, 1)
        self.padding = (kernel//2, 1) if H else (1, kernel//2)
        self.group = group
        self.pad = nn.ReflectionPad2d((pad, pad, 0, 0)) if H else nn.ReflectionPad2d((0, 0, pad, pad))
        self.conv = nn.Conv2d(dim, group*kernel, kernel_size=1, stride=1, bias=False)
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.filter_act = nn.Sigmoid()

    def forward(self, x):
        filter = self.ap(x)
        filter = self.conv(filter)
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c//self.group, self.k, h*w)
        n, c1, p, q = filter.shape
        filter = filter.reshape(n, c1//self.k, self.k, p*q).unsqueeze(2)
        filter = self.filter_act(filter)
        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out

# Global Pool Strip Attention
class GlobalPoolStripAttention(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.channel = k
        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_pool = nn.AdaptiveAvgPool2d((1, None))
        self.hori_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x):
        hori_l = self.hori_pool(x)
        hori_h = x - hori_l
        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h
        vert_l = self.vert_pool(hori_out)
        vert_h = hori_out - vert_l
        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h
        return x * self.beta + vert_out * self.gamma

# Local Pool Strip Attention
class LocalPoolStripAttention(nn.Module):
    def __init__(self, k, kernel=7):
        super().__init__()
        self.channel = k
        self.vert_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_high = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_low = nn.Parameter(torch.zeros(k, 1, 1))
        self.hori_high = nn.Parameter(torch.zeros(k, 1, 1))
        self.vert_pool = nn.AvgPool2d(kernel_size=(kernel, 1), stride=1)
        self.hori_pool = nn.AvgPool2d(kernel_size=(1, kernel), stride=1)
        pad_size = kernel // 2
        self.pad_vert = nn.ReflectionPad2d((0, 0, pad_size, pad_size))
        self.pad_hori = nn.ReflectionPad2d((pad_size, pad_size, 0, 0))
        self.gamma = nn.Parameter(torch.zeros(k, 1, 1))
        self.beta = nn.Parameter(torch.ones(k, 1, 1))

    def forward(self, x):
        hori_l = self.hori_pool(self.pad_hori(x))
        hori_h = x - hori_l
        hori_out = self.hori_low * hori_l + (self.hori_high + 1.) * hori_h
        vert_l = self.vert_pool(self.pad_vert(hori_out))
        vert_h = hori_out - vert_l
        vert_out = self.vert_low * vert_l + (self.vert_high + 1.) * vert_h
        return x * self.beta + vert_out * self.gamma

# Spectral Attention Module
class SpecAtte(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.global_att = GlobalPoolStripAttention(k)
        self.local_att_7 = LocalPoolStripAttention(k, kernel=7)
        self.local_att_11 = LocalPoolStripAttention(k, kernel=11)
        self.conv = nn.Conv2d(k, k, 1)

    def forward(self, x):
        global_out = self.global_att(x)
        local_7_out = self.local_att_7(x)
        local_11_out = self.local_att_11(x)
        out = global_out + local_7_out + local_11_out
        return self.conv(out)

# DSANet Model
class DSANet(nn.Module):
    def __init__(self, num_res=4):
        super(DSANet, self).__init__()
        base_channel = 32
        self.Encoder = nn.ModuleList([
            EBlock(base_channel, num_res),
            EBlock(base_channel*2, num_res),
            EBlock(base_channel*4, num_res),
        ])
        self.feat_extract = nn.ModuleList([
            BasicConv(3, base_channel, kernel_size=3, relu=True, stride=1),
            BasicConv(base_channel, base_channel*2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*2, base_channel*4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_channel*4, base_channel*2, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel*2, base_channel, kernel_size=4, relu=True, stride=2, transpose=True),
            BasicConv(base_channel, 3, kernel_size=3, relu=False, stride=1)
        ])
        self.Decoder = nn.ModuleList([
            DBlock(base_channel * 4, num_res),
            DBlock(base_channel * 2, num_res),
            DBlock(base_channel, num_res)
        ])
        self.Convs = nn.ModuleList([
            BasicConv(base_channel * 4, base_channel * 2, kernel_size=1, relu=True, stride=1),
            BasicConv(base_channel * 2, base_channel, kernel_size=1, relu=True, stride=1),
        ])
        self.ConvsOut = nn.ModuleList([
            BasicConv(base_channel * 4, 3, kernel_size=3, relu=False, stride=1),
            BasicConv(base_channel * 2, 3, kernel_size=3, relu=False, stride=1),
        ])
        self.FAM1 = FAM(base_channel * 4)
        self.SCM1 = SCM(base_channel * 4)
        self.FAM2 = FAM(base_channel * 2)
        self.SCM2 = SCM(base_channel * 2)

    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_4 = F.interpolate(x_2, scale_factor=0.5, mode='bilinear', align_corners=False)
        z2 = self.SCM2(x_2)
        z4 = self.SCM1(x_4)
        outputs = []
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)
        z = self.feat_extract[1](res1)
        z = self.FAM2(z, z2)
        res2 = self.Encoder[1](z)
        z = self.feat_extract[2](res2)
        z = self.FAM1(z, z4)
        z = self.Encoder[2](z)
        z = self.Decoder[0](z)
        z_ = self.ConvsOut[0](z)
        z = self.feat_extract[3](z)
        z_ = F.interpolate(z_, size=(x_4.shape[2], x_4.shape[3]), mode='bilinear', align_corners=False)
        outputs.append(z_ + x_4)
        z = torch.cat([z, res2], dim=1)
        z = self.Convs[0](z)
        z = self.Decoder[1](z)
        z_ = self.ConvsOut[1](z)
        z = self.feat_extract[4](z)
        z_ = F.interpolate(z_, size=(x_2.shape[2], x_2.shape[3]), mode='bilinear', align_corners=False)
        outputs.append(z_ + x_2)
        z = torch.cat([z, res1], dim=1)
        z = self.Convs[1](z)
        z = self.Decoder[2](z)
        z = self.feat_extract[5](z)
        outputs.append(z + x)
        return outputs

# Dataset for Dehazing
class DehazeDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.hazy_images = sorted([f for f in os.listdir(data_dir) if not f.endswith('_gt.png')])

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_path = os.path.join(self.data_dir, self.hazy_images[idx])
        clean_path = os.path.join(self.data_dir, self.hazy_images[idx].replace('.png', '_gt.png'))
        input_img = cv2.imread(hazy_path, cv2.IMREAD_COLOR)
        clean_img = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) / 255.0
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB) / 255.0
        
        if self.train:
            input_img = cv2.resize(input_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            clean_img = cv2.resize(clean_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = input_img.shape[:2]
            h = h - (h % 4)  # Ensure divisible by 4
            w = w - (w % 4)
            input_img = cv2.resize(input_img, (w, h), interpolation=cv2.INTER_LINEAR)
            clean_img = cv2.resize(clean_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        input_tensor = torch.from_numpy(input_img.transpose(2, 0, 1)).float()
        clean_tensor = torch.from_numpy(clean_img.transpose(2, 0, 1)).float()
        return input_tensor, clean_tensor
    
# Training and Evaluation with Multiscale Loss
def train_and_evaluate(data_dir, num_epochs=10, batch_size=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = DSANet().to(device)
    criterion = nn.MSELoss()  # Use MSE loss
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
            for input_img, clean_img in train_loader:
                input_img, clean_img = input_img.to(device), clean_img.to(device)
                optimizer.zero_grad()
                output_imgs = model(input_img)  # Outputs: [64x64, 128x128, 256x256]
                clean_img2 = F.interpolate(clean_img, scale_factor=0.5, mode='bilinear', align_corners=False)
                clean_img4 = F.interpolate(clean_img, scale_factor=0.25, mode='bilinear', align_corners=False)
                l1 = criterion(output_imgs[0], clean_img4)
                l2 = criterion(output_imgs[1], clean_img2)
                l3 = criterion(output_imgs[2], clean_img)
                loss = l1 + l2 + l3
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'Loss': train_loss / (pbar.n + 1)})
                pbar.update(1)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

    print("Final evaluation on validation set with original GT resolution...")
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for input_img, clean_img in tqdm(val_loader, desc="Final Evaluation", unit="image"):
            input_img, clean_img = input_img.to(device), clean_img.to(device)
            output_imgs = model(input_img)
            output_img = output_imgs[-1]  # Take highest resolution output
            output_img = F.interpolate(output_img, size=(clean_img.shape[2], clean_img.shape[3]), mode='bilinear', align_corners=False)
            output_np = output_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            clean_np = clean_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr_values.append(psnr(clean_np, output_np, data_range=1.0))
            ssim_values.append(ssim(clean_np, output_np, multichannel=True, data_range=1.0, win_size=11, channel_axis=2))

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Final Average PSNR: {avg_psnr:.4f}")
    print(f"Final Average SSIM: {avg_ssim:.4f}")

    # Plot PSNR and SSIM
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(psnr_values) + 1), psnr_values, marker='o', label='PSNR')
    plt.axhline(y=avg_psnr, color='r', linestyle='--', label=f'Avg PSNR: {avg_psnr:.2f}')
    plt.xlabel('Image Index')
    plt.ylabel('PSNR')
    plt.title('PSNR Across Validation Images')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(ssim_values) + 1), ssim_values, marker='o', label='SSIM')
    plt.axhline(y=avg_ssim, color='r', linestyle='--', label=f'Avg SSIM: {avg_ssim:.2f}')
    plt.xlabel('Image Index')
    plt.ylabel('SSIM')
    plt.title('SSIM Across Validation Images')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), 'dsanet_trained.pth')
    print("Model saved as 'dsanet_trained.pth'")

# Run the training
if __name__ == "__main__":
    data_dir = 'path to your dataset'
    train_and_evaluate(data_dir, num_epochs=10, batch_size=2)
