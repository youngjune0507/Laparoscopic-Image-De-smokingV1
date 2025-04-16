#source : https://github.com/c-yn/OKNet/blob/main/Dehazing/ITS/models/OKNet.py


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
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x

# Encoder Block
class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()
        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Decoder Block
class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()
        layers = [ResBlock(channel, channel) for _ in range(num_res)]
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

# Bottleneck Module
class BottleNect(nn.Module):
    def __init__(self, dim):
        super().__init__()
        ker = 63
        pad = ker // 2
        self.in_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1),
            nn.GELU()
        )
        self.out_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1)
        self.dw_13 = nn.Conv2d(dim, dim, kernel_size=(1, ker), padding=(0, pad), stride=1, groups=dim)
        self.dw_31 = nn.Conv2d(dim, dim, kernel_size=(ker, 1), padding=(pad, 0), stride=1, groups=dim)
        self.dw_33 = nn.Conv2d(dim, dim, kernel_size=ker, padding=pad, stride=1, groups=dim)
        self.dw_11 = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=dim)
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fac_conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.fac_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fgm = FGM(dim)

    def forward(self, x):
        out = self.in_conv(x)
        x_att = self.fac_conv(self.fac_pool(out))
        x_fft = torch.fft.fft2(out, norm='backward')
        x_fft = x_att * x_fft
        x_fca = torch.fft.ifft2(x_fft, dim=(-2, -1), norm='backward')
        x_fca = torch.abs(x_fca)
        x_att = self.conv(self.pool(x_fca))
        x_sca = x_att * x_fca
        x_sca = self.fgm(x_sca)
        out = x + self.dw_13(out) + self.dw_31(out) + self.dw_33(out) + self.dw_11(out) + x_sca
        out = self.act(out)
        return self.out_conv(out)

# Frequency Gating Module
class FGM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim*2, 3, 1, 1)
        self.dwconv1 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.dwconv2 = nn.Conv2d(dim, dim, 1, 1, groups=1)
        self.alpha = nn.Parameter(torch.zeros(dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(dim, 1, 1))

    def forward(self, x):
        fft_size = x.size()[2:]
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x)
        x2_fft = torch.fft.fft2(x2, norm='backward')
        out = x1 * x2_fft
        out = torch.fft.ifft2(out, dim=(-2, -1), norm='backward')
        out = torch.abs(out)
        return out * self.alpha + x * self.beta

# Feature Aggregation Module
class FAM(nn.Module):
    def __init__(self, channel):
        super(FAM, self).__init__()
        self.merge = BasicConv(channel*2, channel, kernel_size=3, stride=1, relu=False)

    def forward(self, x1, x2):
        return self.merge(torch.cat([x1, x2], dim=1))

# OKNet Model
class OKNet(nn.Module):
    def __init__(self, num_res=4):
        super(OKNet, self).__init__()
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
        self.bottelneck = BottleNect(base_channel * 4)

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
        z = self.bottelneck(z)
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
        gt_path = os.path.join(self.data_dir, self.hazy_images[idx].replace('.png', '_gt.png'))
        hazy_img = cv2.imread(hazy_path, cv2.IMREAD_COLOR)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB) / 255.0
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) / 255.0
        
        if self.train:
            # Resize to fixed size during training
            hazy_img = cv2.resize(hazy_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            gt_img = cv2.resize(gt_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        else:
            # Adjust to multiple of 4 for validation
            h, w = hazy_img.shape[:2]
            h = h - (h % 4)
            w = w - (w % 4)
            hazy_img = cv2.resize(hazy_img, (w, h), interpolation=cv2.INTER_LINEAR)
            gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        hazy_img = torch.from_numpy(hazy_img.transpose(2, 0, 1)).float()
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).float()
        return hazy_img, gt_img

# Function to compute FFT for frequency domain loss
def compute_fft(image):
    # Compute 2D FFT for each channel
    fft = torch.fft.fft2(image, dim=(-2, -1))
    fft = torch.stack((fft.real, fft.imag), dim=-1)  # [N, C, H, W, 2]
    return fft

# Training and Evaluation with Dual Domain L1 Loss
def train_and_evaluate(data_dir, num_epochs=15, batch_size=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Initialize model
    model = OKNet().to(device)
    criterion = nn.L1Loss()  # Use L1 loss for both spatial and frequency domains
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Prepare dataset and data loaders
    train_dataset = DehazeDataset(data_dir, train=True)
    val_dataset = DehazeDataset(data_dir, train=False)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, _ = random_split(train_dataset, [train_size, val_size])
    _, val_dataset = random_split(val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for hazy_img, gt_img in train_loader:
                hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
                optimizer.zero_grad()
                dehaze_outputs = model(hazy_img)  # Outputs: [64x64, 128x128, 256x256]
                
                # Prepare multi-scale GT images
                gt_img2 = F.interpolate(gt_img, scale_factor=0.5, mode='bilinear', align_corners=False)
                gt_img4 = F.interpolate(gt_img, scale_factor=0.25, mode='bilinear', align_corners=False)
                
                # Spatial domain L1 loss
                spatial_loss = (criterion(dehaze_outputs[0], gt_img4) + 
                               criterion(dehaze_outputs[1], gt_img2) + 
                               criterion(dehaze_outputs[2], gt_img))
                
                # Frequency domain L1 loss
                fft_out0 = compute_fft(dehaze_outputs[0])
                fft_gt4 = compute_fft(gt_img4)
                fft_out1 = compute_fft(dehaze_outputs[1])
                fft_gt2 = compute_fft(gt_img2)
                fft_out2 = compute_fft(dehaze_outputs[2])
                fft_gt = compute_fft(gt_img)
                
                freq_loss = (criterion(fft_out0, fft_gt4) + 
                            criterion(fft_out1, fft_gt2) + 
                            criterion(fft_out2, fft_gt))
                
                # Combine losses: L_s + 0.1 * L_f
                loss = spatial_loss + 0.1 * freq_loss
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'Loss': train_loss / (pbar.n + 1)})
                pbar.update(1)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

    # Final evaluation
    print("Final evaluation on validation set with original GT resolution...")
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for hazy_img, gt_img in tqdm(val_loader, desc="Final Evaluation", unit="image"):
            hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
            dehaze_outputs = model(hazy_img)
            dehaze_img = dehaze_outputs[-1]  # Take highest resolution output
            dehaze_img = F.interpolate(dehaze_img, size=(gt_img.shape[2], gt_img.shape[3]), mode='bilinear', align_corners=False)
            dehaze_np = dehaze_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = gt_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr_values.append(psnr(gt_np, dehaze_np, data_range=1.0))
            ssim_values.append(ssim(gt_np, dehaze_np, multichannel=True, data_range=1.0, win_size=11, channel_axis=2))

    # Print results
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Final Average PSNR: {avg_psnr:.4f}")
    print(f"Final Average SSIM: {avg_ssim:.4f}")

    # Visualize results
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

    # Save model
    torch.save(model.state_dict(), 'oknet_trained.pth')
    print("Model saved as 'oknet_trained.pth'")

# Execution
if __name__ == "__main__":
    data_dir = 'path to your dataset'
    train_and_evaluate(data_dir, num_epochs=15, batch_size=8)
