import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
from numpy.random import default_rng
import math
from PIL import Image

from datasets import load_dataset
from torchvision.transforms import v2
from datasets import DatasetDict

import torch
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision import models

import einops
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.05, style_weight=0.0001):
        super(CombinedLoss, self).__init__()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight

        self.l1_loss = nn.L1Loss()
        vgg = vgg19(pretrained=True).features
        self.vgg_layers = vgg[:36].eval()      
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

    def compute_feature_difference(self, prediction, target):
        feature_diff = 0
        for p_f, t_f in zip(prediction, target):
            p_f_norm = F.normalize(p_f, p=2, dim=1)
            t_f_norm = F.normalize(t_f, p=2, dim=1)
            feature_diff += torch.norm(p_f_norm - t_f_norm, p=2)

        return feature_diff

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in {3, 8, 17, 26, 35}:  # conv1_2, conv2_2, conv3_4, conv4_4, conv5_4
                features.append(x)
        return features
    
    def forward(self, prediction, target):
        self.vgg_layers.to(dtype=prediction.dtype, device=prediction.device)

        prediction_features = self.extract_features(prediction)
        target_features = self.extract_features(target)

        # L1 Loss
        l1_loss = self.l1_loss(prediction, target)

        # Perceptual Loss (Feature-Level MSE)
        perceptual_loss = sum(F.mse_loss(p_f, t_f)
                              for p_f, t_f in zip(prediction_features, target_features))

        # Style Loss (Gram Matrix MSE)
        style_loss = self.compute_feature_difference(prediction_features, target_features)


        l1 = self.l1_weight * l1_loss
        perceptual = self.perceptual_weight * perceptual_loss
        style = self.style_weight * style_loss

        return l1, perceptual, style
    
def add_irregular_blobs(image, min_radius, max_radius, num_blobs):
    if image.ndim == 3:
        if image.shape[0] in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = img.astype(np.uint8)

    height, width = img.shape[:2]
    rng = default_rng()
    num_points = 12
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(num_blobs):
        base_radius = rng.integers(min_radius, max_radius)
        center_x = rng.integers(base_radius, width - base_radius)
        center_y = rng.integers(base_radius, height - base_radius)
        contour_points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            radius = base_radius + rng.integers(-5, 5) 
            x = int(center_x + radius * math.cos(angle))
            y = int(center_y + radius * math.sin(angle))
            contour_points.append((x, y))
        
        contour_points = np.array([contour_points], dtype=np.int32)
        cv2.fillPoly(mask, contour_points, 255)

    mask_rgb = cv2.merge([mask, mask, mask])

    result = cv2.add(img, mask_rgb)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    tensor_image = torch.from_numpy(result)
    tensor_image = tensor_image.permute(2, 0, 1)


    return tensor_image


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self.conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return out
    
class Encoder(nn.Module):

    @staticmethod
    def conv_block(in_size: int, out_size: int):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )
    
    @staticmethod
    def conv_block_without_stride(in_size: int, out_size: int):
        return nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )


    def __init__(self, latent_width, **kwargs) -> None:
        super().__init__()

        self.latent_width = latent_width

        self.model = nn.Sequential(
            # Encoder.conv_block(3, 64),
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Encoder.conv_block_without_stride(64, 64),

            # Encoder.conv_block(64, 128),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Encoder.conv_block_without_stride(128, 128),

            Encoder.conv_block(128, 256),
            Encoder.conv_block_without_stride(256, 256),
            Encoder.conv_block(256, 512),
            Encoder.conv_block_without_stride(512, 512),
            Encoder.conv_block_without_stride(512, 128),
            Encoder.conv_block_without_stride(128, 128),
        )

        self.fc = nn.Sequential(
            nn.Linear(128*16*16, self.latent_width), 
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = einops.rearrange(x, "b c w h -> b (c w h)")
        x = self.fc(x)
        return x
    
class Decoder(nn.Module):
    @staticmethod
    def conv_block(in_size: int, out_size: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=3, 
                               stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        )
    
    @staticmethod
    def conv_block_without_stride(in_size: int, out_size: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, kernel_size=3, 
                               stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
        )
    
    def __init__(self, latent_width, **kwargs) -> None:
        super().__init__()

        self.latent_width = latent_width
        self.model = nn.Sequential(
            Decoder.conv_block_without_stride(128, 128),
            # Decoder.conv_block(128, 512),
            nn.ConvTranspose2d(128, 512, kernel_size=3, 
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),


            Decoder.conv_block_without_stride(512, 512),
            # Decoder.conv_block(512, 256),
            nn.ConvTranspose2d(512, 256, kernel_size=3, 
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            Decoder.conv_block_without_stride(256, 256),
            Decoder.conv_block(256, 128),
            Decoder.conv_block_without_stride(128, 128),
            Decoder.conv_block_without_stride(128, 64),
            Decoder.conv_block_without_stride(64, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=3, 
                               stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.latent_width, 128*16*16), 
            nn.Tanh(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = einops.rearrange(x, "b (c w h) -> b c w h", c=128, w=16, h=16)
        x = self.model(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        decoded = self.decoder(latent)

        return decoded, latent

class UNetUpscaler(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetUpscaler, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 256)
        
        # Decoder
        
        self.up3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        
        self.final_up = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final_up(d1)
        out = self.final_conv(out)
        return out


class ClusterNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClusterNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)