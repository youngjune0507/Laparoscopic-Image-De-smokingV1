#source : https://github.com/thuBingo/DehazeNet_Pytorch

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

# ------------------- BRelu Activation -------------------
class BRelu(nn.Hardtanh):
    """Bounded ReLU activation (0 to 1) for GPU compatibility."""
    def __init__(self, inplace=False):
        super(BRelu, self).__init__(0., 1., inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

# ------------------- DehazeNet Model -------------------
class DehazeNet(nn.Module):
    """DehazeNet model for image dehazing with Maxout and BRelu."""
    def __init__(self, input=16, groups=4):
        super(DehazeNet, self).__init__()
        self.input = input
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.input, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=7, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=7, stride=1)
        self.conv5 = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=6)  # Output 3 channels for RGB
        self.brelu = BRelu()
        
        # Initialize weights
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def Maxout(self, x, groups):
        """Maxout operation to select maximum features across groups."""
        x = x.reshape(x.shape[0], groups, x.shape[1] // groups, x.shape[2], x.shape[3])
        x, _ = torch.max(x, dim=2, keepdim=True)
        out = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])
        return out

    def forward(self, x):
        out = self.conv1(x)
        out = self.Maxout(out, self.groups)
        out1 = self.conv2(out)
        out2 = self.conv3(out)
        out3 = self.conv4(out)
        y = torch.cat((out1, out2, out3), dim=1)
        y = self.maxpool(y)
        y = self.conv5(y)
        y = self.brelu(y)
        return y  # Output shape: [bs, 3, H-18, W-18]

# ------------------- Dataset -------------------
class DehazeDataset(Dataset):
    """
    Dataset class for loading Hazy and Ground Truth images from the Desmoke dataset.
    Resizes training images to 256x256 and adjusts validation images to multiples of 4.
    """
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
        if hazy_img is None or gt_img is None:
            raise ValueError(f"Failed to load images: {hazy_path}, {gt_path}")
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB) / 255.0
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) / 255.0
        
        if self.train:
            hazy_img = cv2.resize(hazy_img, (256, 256), interpolation=cv2.INTER_LINEAR)
            gt_img = cv2.resize(gt_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        else:
            h, w = hazy_img.shape[:2]
            h = h - (h % 4)
            w = w - (w % 4)
            hazy_img = cv2.resize(hazy_img, (w, h), interpolation=cv2.INTER_LINEAR)
            gt_img = cv2.resize(gt_img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        hazy_img = torch.from_numpy(hazy_img.transpose(2, 0, 1)).float()
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).float()
        return hazy_img, gt_img

# ------------------- Training and Evaluation -------------------
def train_and_evaluate(data_dir, num_epochs=20, batch_size=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train and evaluate DehazeNet on the Desmoke dataset.
    Uses MSE Loss with gradient clipping (max_norm=0.1).
    Resizes training images to 256x256 and adjusts validation images to multiples of 4.
    Evaluates with MSE Loss, PSNR, and SSIM, and visualizes PSNR/SSIM plots.

    Args:
        data_dir (str): Path to the Desmoke dataset.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        device (str): Device to run the model on ('cuda' or 'cpu').
    """
    # Initialize model
    model = DehazeNet(input=16, groups=4).to(device)
    
    # Define loss function and optimizer
    mse_loss = nn.MSELoss().to(device)
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
                
                # Resize dehaze_img to match gt_img due to model output size reduction
                dehaze_img = F.interpolate(dehaze_img, size=gt_img.shape[2:], mode='bilinear', align_corners=False)
                
                # Compute MSE loss
                loss = mse_loss(dehaze_img, gt_img)
                
                loss.backward()
                # Apply gradient clipping (norm constrained to 0.1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'MSE Loss': train_loss / (pbar.n + 1)})
                pbar.update(1)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training MSE Loss: {avg_train_loss:.4f}")

        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for hazy_img, gt_img in val_loader:
                hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
                dehaze_img = model(hazy_img)
                # Resize dehaze_img to match gt_img
                dehaze_img = F.interpolate(dehaze_img, size=gt_img.shape[2:], mode='bilinear', align_corners=False)
                loss = mse_loss(dehaze_img, gt_img)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation MSE Loss: {avg_val_loss:.4f}")

    # Final evaluation
    print("Final evaluation on validation set with adjusted GT resolution...")
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for hazy_img, gt_img in tqdm(val_loader, desc="Final Evaluation", unit="image"):
            hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
            dehaze_img = model(hazy_img)
            
            # Resize dehaze_img to GT resolution
            dehaze_img = F.interpolate(dehaze_img, size=gt_img.shape[2:], mode='bilinear', align_corners=False)
            
            # Verify resolution match
            assert dehaze_img.shape[2:] == gt_img.shape[2:], f"Resolution mismatch: dehaze {dehaze_img.shape[2:]} vs GT {gt_img.shape[2:]}"
            
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
    plt.savefig('psnr_ssim_plot.png', dpi=300)
    plt.close()

    # Save model
    torch.save(model.state_dict(), 'dehazenet_trained.pth')
    print("Model saved as 'dehazenet_trained.pth'")


# ------------------- Main Execution -------------------
if __name__ == "__main__":
    data_dir = 'path to your dataset'
    train_and_evaluate(data_dir, num_epochs=100, batch_size=128)
