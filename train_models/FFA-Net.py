#source : https://github.com/zhilin007/FFA-Net/blob/master/net/models/FFA.py
# FFA-Net for Image Dehazing with Dual Domain L1 Loss

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

# Default Convolution
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

# Pixel Attention Layer
class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

# Residual Block
class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

# Group of Blocks
class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)
    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

# FFA-Net Model
class FFA(nn.Module):
    def __init__(self, gps=3, blocks=19, conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps == 3
        self.g1 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size, blocks=blocks)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim * self.gps, self.dim // 16, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim // 16, self.dim * self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.palayer = PALayer(self.dim)
        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)
        ]
        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1, res2, res3], dim=1))
        w = w.view(-1, self.gps, self.dim)[:, :, :, None, None]
        out = w[:, 0, :] * res1 + w[:, 1, :] * res2 + w[:, 2, :] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1

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
def train_and_evaluate(data_dir, num_epochs=20, batch_size=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Initialize model
    model = FFA(gps=3, blocks=19).to(device)
    
    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Prepare dataset
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
                dehaze_img = model(hazy_img)
                
                # Spatial domain L1 loss
                spatial_loss = criterion(dehaze_img, gt_img)
                
                # Frequency domain L1 loss
                fft_dehaze = compute_fft(dehaze_img)
                fft_gt = compute_fft(gt_img)
                freq_loss = criterion(fft_dehaze, fft_gt)
                
                # Combine losses: L_s + 0.1 * L_f
                loss = spatial_loss + 0.1 * freq_loss
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'Total Loss': train_loss / (pbar.n + 1)})
                pbar.update(1)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for hazy_img, gt_img in val_loader:
                hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
                dehaze_img = model(hazy_img)
                
                # Spatial domain L1 loss
                spatial_loss = criterion(dehaze_img, gt_img)
                
                # Frequency domain L1 loss
                fft_dehaze = compute_fft(dehaze_img)
                fft_gt = compute_fft(gt_img)
                freq_loss = criterion(fft_dehaze, fft_gt)
                
                # Combine losses
                loss = spatial_loss + 0.1 * freq_loss
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

    # Final evaluation
    print("Final evaluation on validation set with original GT resolution...")
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for hazy_img, gt_img in tqdm(val_loader, desc="Final Evaluation", unit="image"):
            hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
            dehaze_img = model(hazy_img)
            
            # Resize to GT resolution if necessary
            dehaze_img = F.interpolate(dehaze_img, size=gt_img.shape[2:], mode='bilinear', align_corners=False)
            
            dehaze_np = dehaze_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = gt_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            psnr_values.append(psnr(gt_np, dehaze_np, data_range=1.0))
            ssim_values.append(ssim(gt_np, dehaze_np, multichannel=True, data_range=1.0, win_size=11, channel_axis=2))

    # Print results
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    print(f"Final Average PSNR: {avg_psnr:.4f}")
    print(f"Final Average SSIM: {avg_ssim:.4f}")

    # Plot graphs
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
    torch.save(model.state_dict(), 'ffa_net_trained.pth')
    print("Model saved as 'ffa_net_trained.pth'")

# Execution
if __name__ == "__main__":
    data_dir = 'path to your dataset'
    train_and_evaluate(data_dir, num_epochs=20, batch_size=2)
