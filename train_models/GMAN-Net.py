# Source: https://github.com/Seanforfun/GMAN_Net_Haze_Removal/blob/master/GMEAN_NET/gman_model.py

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
from torchvision.models import vgg16

# VGG16-based Perceptual Loss
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = vgg16(pretrained=True).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg_layers = vgg
        self.layer_names = ['15', '22']  # conv3_3, conv4_3
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        loss = 0
        x_vgg, y_vgg = x, y
        for i, layer in enumerate(self.vgg_layers):
            x_vgg = layer(x_vgg)
            y_vgg = layer(y_vgg)
            if str(i) in self.layer_names:
                loss += self.criterion(x_vgg, y_vgg)
        return loss

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=2):
        super(ResidualBlock, self).__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.block = nn.Sequential(*layers)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        return F.relu(out + identity)

# GMAN_V1 Model
class GMAN_V1(nn.Module):
    def __init__(self):
        super(GMAN_V1, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.down1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(128, 64, num_layers=2)
        self.res2 = ResidualBlock(64, 64, num_layers=2)
        self.res3 = ResidualBlock(64, 64, num_layers=5)
        self.res4 = ResidualBlock(64, 64, num_layers=5)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x_s = x
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = F.relu(self.up1(x))
        x = F.relu(self.up2(x))
        x_r = self.conv_out(x)
        x_r = F.interpolate(x_r, size=(x_s.shape[2], x_s.shape[3]), mode='bilinear', align_corners=False)
        x_r = x_r + x_s
        x_r = F.relu(x_r)
        return x_r

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
        
        hazy_img = torch.from_numpy(hazy_img.transpose(2, 0, 1)).float()
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).float()
        return hazy_img, gt_img

# Training and Evaluation with MSE and Perceptual Loss
def train_and_evaluate(data_dir, num_epochs=10, batch_size=8, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Initialize model
    model = GMAN_V1().to(device)
    mse_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss(device)
    
    # Define optimizer
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
                dehaze_img = model(hazy_img)
                
                # Compute MSE loss
                mse = mse_loss(dehaze_img, gt_img)
                
                # Compute perceptual loss
                percep = perceptual_loss(dehaze_img, gt_img)
                
                # Combine losses: MSE + 0.1 * Perceptual
                loss = mse + 0.1 * percep
                
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
            dehaze_img = model(hazy_img)
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
    torch.save(model.state_dict(), 'gman_v1_trained.pth')
    print("Model saved as 'gman_v1_trained.pth'")

# Execution
if __name__ == "__main__":
    data_dir = 'path to your dataset'
    train_and_evaluate(data_dir, num_epochs=10, batch_size=8)
