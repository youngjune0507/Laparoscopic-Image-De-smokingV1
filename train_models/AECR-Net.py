#source : https://github.com/GlassyWu/AECR-Net

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import vgg16
from torchvision.ops import DeformConv2d
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import math

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

def isqrt_newton_schulz_autograd(A, numIters):
    dim = A.shape[0]
    normA = A.norm()
    Y = A.div(normA)
    I = torch.eye(dim, dtype=A.dtype, device=A.device)
    Z = torch.eye(dim, dtype=A.dtype, device=A.device)
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z @ Y)
        Y = Y @ T
        Z = T @ Z
    A_isqrt = Z / torch.sqrt(normA)
    return A_isqrt

def isqrt_newton_schulz_autograd_batch(A, numIters):
    batchSize, dim, _ = A.shape
    normA = A.view(batchSize, -1).norm(2, 1).view(batchSize, 1, 1)
    Y = A.div(normA)
    I = torch.eye(dim, dtype=A.dtype, device=A.device).unsqueeze(0).expand_as(A)
    Z = torch.eye(dim, dtype=A.dtype, device=A.device).unsqueeze(0).expand_as(A)
    for i in range(numIters):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    A_isqrt = Z / torch.sqrt(normA)
    return A_isqrt

# ------------------- FastDeconv Module -------------------
class FastDeconv(nn.Module):
    """Fast Deconvolution layer to normalize feature correlations."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, eps=1e-5, n_iter=5, momentum=0.1, block=64, sampling_stride=3, freeze=False, freeze_iter=100):
        super(FastDeconv, self).__init__()
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter = 0
        self.track_running_stats = True
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

        if block > in_channels:
            block = in_channels
        else:
            if in_channels % block != 0:
                block = math.gcd(block, in_channels)
        if groups > 1:
            block = in_channels // groups
        self.block = block
        self.num_features = kernel_size ** 2 * block
        if groups == 1:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_deconv', torch.eye(self.num_features))
        else:
            self.register_buffer('running_mean', torch.zeros(kernel_size ** 2 * in_channels))
            self.register_buffer('running_deconv', torch.eye(self.num_features).repeat(in_channels // block, 1, 1))
        self.sampling_stride = sampling_stride * stride
        self.freeze_iter = freeze_iter
        self.freeze = freeze

    def forward(self, x):
        N, C, H, W = x.shape
        B = self.block
        frozen = self.freeze and (self.counter > self.freeze_iter)
        if self.training and self.track_running_stats:
            self.counter += 1
            self.counter %= (self.freeze_iter * 10)

        if self.training and not frozen:
            if self.kernel_size[0] > 1:
                X = F.unfold(x, self.kernel_size, self.dilation, self.padding, self.sampling_stride).transpose(1, 2).contiguous()
            else:
                X = x.permute(0, 2, 3, 1).contiguous().view(-1, C)[::self.sampling_stride ** 2, :]
            if self.groups == 1:
                X = X.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features)
            else:
                X = X.view(-1, X.shape[-1])
            X_mean = X.mean(0)
            X = X - X_mean.unsqueeze(0)
            if self.groups == 1:
                Id = torch.eye(X.shape[1], dtype=X.dtype, device=X.device)
                Cov = torch.addmm(self.eps, Id, 1. / X.shape[0], X.t(), X)
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            else:
                X = X.view(-1, self.groups, self.num_features).transpose(0, 1)
                Id = torch.eye(self.num_features, dtype=X.dtype, device=X.device).expand(self.groups, self.num_features, self.num_features)
                Cov = torch.baddbmm(self.eps, Id, 1. / X.shape[1], X.transpose(1, 2), X)
                deconv = isqrt_newton_schulz_autograd_batch(Cov, self.n_iter)
            if self.track_running_stats:
                self.running_mean.mul_(1 - self.momentum)
                self.running_mean.add_(X_mean.detach() * self.momentum)
                self.running_deconv.mul_(1 - self.momentum)
                self.running_deconv.add_(deconv.detach() * self.momentum)
        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        if self.groups == 1:
            w = self.weight.view(-1, self.num_features, C // B).transpose(1, 2).contiguous().view(-1, self.num_features) @ deconv
            b = self.bias - (w @ (X_mean.unsqueeze(1))).view(self.weight.shape[0], -1).sum(1) if self.bias is not None else None
            w = w.view(-1, C // B, self.num_features).transpose(1, 2).contiguous()
        else:
            w = self.weight.view(C // B, -1, self.num_features) @ deconv
            b = self.bias - (w @ (X_mean.view(-1, self.num_features, 1))).view(self.bias.shape) if self.bias is not None else None
        w = w.view(self.weight.shape)
        x = F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)
        return x

# ------------------- AECR-Net Components -------------------
class PALayer(nn.Module):
    """Pixel Attention Layer to enhance feature importance."""
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

class CALayer(nn.Module):
    """Channel Attention Layer to reweight channel importance."""
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

class DehazeBlock(nn.Module):
    """Dehazing block with residual connections, CA, and PA layers."""
    def __init__(self, conv, dim, kernel_size):
        super(DehazeBlock, self).__init__()
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

class DCNBlock(nn.Module):
    """Deformable Convolution Block for flexible feature alignment."""
    def __init__(self, in_channel, out_channel):
        super(DCNBlock, self).__init__()
        self.offset_conv = nn.Conv2d(in_channel, 2 * 3 * 3, kernel_size=3, padding=1, bias=True)
        self.dcn = DeformConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        offset = self.offset_conv(x)
        return self.dcn(x, offset)

class Mix(nn.Module):
    """Learnable feature mixing layer with sigmoid-weighted blending."""
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

# ------------------- AECR-Net Model -------------------
class Dehaze(nn.Module):
    """
    AECR-Net model for image dehazing, based on FFA-Net with deformable convolutions.
    Adjusted output_padding to fix size mismatch in upsampling.
    Reference: https://arxiv.org/abs/2107.10846
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, use_dropout=False, padding_type='reflect'):
        super(Dehaze, self).__init__()
        self.down1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.ReLU(True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.block = DehazeBlock(default_conv, ngf * 4, 3)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=0),  # Adjusted
            nn.ReLU(True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=0),  # Adjusted
            nn.ReLU(True)
        )
        self.up3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        )
        self.dcn_block = DCNBlock(256, 256)
        self.deconv = FastDeconv(3, 3, kernel_size=3, stride=1, padding=1)
        self.mix1 = Mix(m=-1)
        self.mix2 = Mix(m=-0.6)

    def forward(self, input):
        x_deconv = self.deconv(input)
        x_down1 = self.down1(x_deconv)
        x_down2 = self.down2(x_down1)
        x_down3 = self.down3(x_down2)
        x1 = self.block(x_down3)
        x2 = self.block(x1)
        x3 = self.block(x2)
        x4 = self.block(x3)
        x5 = self.block(x4)
        x6 = self.block(x5)
        x_dcn1 = self.dcn_block(x6)
        x_dcn2 = self.dcn_block(x_dcn1)
        x_out_mix = self.mix1(x_down3, x_dcn2)
        x_up1 = self.up1(x_out_mix)
        # Ensure x_up1 matches x_down2 size
        x_up1 = F.interpolate(x_up1, size=(x_down2.size(2), x_down2.size(3)), mode='bilinear', align_corners=False)
        x_up1_mix = self.mix2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_mix)
        out = self.up3(x_up2)
        return out

# ------------------- Loss Functions -------------------
class L1Loss(nn.Module):
    """L1 Loss for pixel-wise difference between Dehazed and Ground Truth images."""
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, x, y):
        return F.l1_loss(x, y)

class PerceptualLoss(nn.Module):
    """
    Contrastive Perceptual Loss using VGG16 features, with normalization by Hazy-Dehazed distance.
    Implements the second term of the AECR-Net loss function, weighted by beta=0.1.
    """
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_4'], weights=None, beta=0.1):
        super(PerceptualLoss, self).__init__()
        self.beta = beta
        vgg = vgg16(pretrained=True).features.eval()
        self.vgg_layers = nn.ModuleList(vgg)
        self.layer_names = layers
        self.weights = weights if weights else [1.0 / len(self.layer_names)] * len(self.layer_names)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def get_features(self, x):
        """Extract features from specified VGG layers."""
        features = []
        layer_idx = 0
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if layer_idx < len(self.layer_names) and isinstance(layer, nn.Conv2d):
                if i in [2, 7, 14]:  # conv1_2, conv2_2, conv3_4
                    features.append(x)
                    layer_idx += 1
        return features

    def forward(self, hazy, dehazed, gt):
        hazy = (hazy - self.mean) / self.std
        dehazed = (dehazed - self.mean) / self.std
        gt = (gt - self.mean) / self.std
        hazy_feats = self.get_features(hazy)
        dehazed_feats = self.get_features(dehazed)
        gt_feats = self.get_features(gt)
        loss = 0
        for i, (gt_f, dehazed_f, hazy_f) in enumerate(zip(gt_feats, dehazed_feats, hazy_feats)):
            gt_dehazed_dist = F.mse_loss(gt_f, dehazed_f)
            hazy_dehazed_dist = F.mse_loss(hazy_f, dehazed_f)
            hazy_dehazed_dist = torch.clamp(hazy_dehazed_dist, min=1e-8)
            loss += self.weights[i] * (gt_dehazed_dist / hazy_dehazed_dist)
        return self.beta * loss

# ------------------- Dataset -------------------
class DehazeDataset(Dataset):
    """
    Dataset class for loading Hazy and Ground Truth images from the Desmoke dataset.
    Pads images to 352x704 to ensure compatibility with AECR-Net.
    Maintains original resolution for visualization.
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
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB) / 255.0
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) / 255.0
        H_orig, W_orig = hazy_img.shape[:2]
        # Pad to 352x704
        pad_h = 352 - H_orig if H_orig < 352 else 0
        pad_w = 704 - W_orig if W_orig < 704 else 0
        if pad_h > 0 or pad_w > 0:
            hazy_img = np.pad(hazy_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            gt_img = np.pad(gt_img, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        hazy_img = torch.from_numpy(hazy_img.transpose(2, 0, 1)).float()
        gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1)).float()
        return hazy_img, gt_img, hazy_path, H_orig, W_orig

# ------------------- Training and Evaluation -------------------
def train_and_evaluate(data_dir, num_epochs=10, batch_size=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train and evaluate the AECR-Net model for image dehazing.
    Uses L1 Loss and Contrastive Perceptual Loss (weighted by 0.1).
    Handles padded inputs (352x704) and restores original dimensions for output.
    Computes PSNR/SSIM metrics for validation.
    
    Args:
        data_dir (str): Path to the Desmoke dataset.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training (set to 1 for variable resolutions).
        device (str): Device to run the model on ('cuda' or 'cpu').
    """
    model = Dehaze(input_nc=3, output_nc=3, ngf=64).to(device)
    l1_loss = L1Loss()
    perceptual_loss = PerceptualLoss(layers=['conv1_2', 'conv2_2', 'conv3_4'], beta=0.1).to(device)
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
            for hazy_img, gt_img, _, H_orig, W_orig in train_loader:
                hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
                optimizer.zero_grad()
                dehaze_img = model(hazy_img)
                # Crop to original dimensions
                dehaze_img = dehaze_img[:, :, :H_orig[0], :W_orig[0]]
                gt_img = gt_img[:, :, :H_orig[0], :W_orig[0]]
                hazy_img = hazy_img[:, :, :H_orig[0], :W_orig[0]]
                l1 = l1_loss(dehaze_img, gt_img)
                p_loss = perceptual_loss(hazy_img, dehaze_img, gt_img)
                loss = l1 + p_loss  # Perceptual loss includes beta=0.1
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({'Loss': train_loss / (pbar.n + 1)})
                pbar.update(1)
        avg_train_loss = train_loss / len(train_loader)
        print(f"Average Training Loss: {avg_train_loss:.4f}")

    print("Generating comparison images and computing metrics for validation set...")
    model.eval()
    os.makedirs('comparisons', exist_ok=True)
    psnr_scores = []
    ssim_scores = []
    with torch.no_grad():
        for idx, (hazy_img, gt_img, hazy_path, H_orig, W_orig) in enumerate(tqdm(val_loader, desc="Generating Comparisons", unit="image")):
            hazy_img, gt_img = hazy_img.to(device), gt_img.to(device)
            dehaze_img = model(hazy_img)
            # Crop to original dimensions
            dehaze_img = dehaze_img[:, :, :H_orig[0], :W_orig[0]]
            hazy_img = hazy_img[:, :, :H_orig[0], :W_orig[0]]
            gt_img = gt_img[:, :, :H_orig[0], :W_orig[0]]
            hazy_np = hazy_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            dehaze_np = dehaze_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gt_np = gt_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Compute PSNR and SSIM
            psnr_score = psnr(gt_np, dehaze_np, data_range=1.0)
            ssim_score = ssim(gt_np, dehaze_np, data_range=1.0, multichannel=True)
            psnr_scores.append(psnr_score)
            ssim_scores.append(ssim_score)

            # Create comparison plot
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(hazy_np)
            plt.title('Hazy Image')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(dehaze_np)
            plt.title(f'Dehazed Image\nPSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}')
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
            print(f"Saved comparison image: {save_path}, PSNR: {psnr_score:.2f}, SSIM: {ssim_score:.4f}")

    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    print(f"Average Validation PSNR: {avg_psnr:.2f}, Average Validation SSIM: {avg_ssim:.4f}")

    torch.save(model.state_dict(), 'aecrnet_trained.pth')
    print("Model saved as 'aecrnet_trained.pth'")



# ------------------- Main Execution -------------------
if __name__ == "__main__":
    data_dir = 'path to your dataset'
    train_and_evaluate(data_dir, num_epochs=10, batch_size=1)
