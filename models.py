import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # Import weights enum


class ResNetCNN_AE(nn.Module):
    def __init__(self):
        super(ResNetCNN_AE, self).__init__()
     
        self.resnet = models.resnet18() 
        
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  
        )
    
    def forward(self, x):
        features = self.resnet(x)
        output = self.decoder(features)
        
        output = torch.sigmoid(output)
        
        return features, output
    

class ImprovedDecoder(nn.Module):
    def __init__(self):
        super(ImprovedDecoder, self).__init__()
        
        self.upscale_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, padding_mode='replicate'), # 'zeros', 'reflect', 'replicate', 'circular'
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.LeakyReLU(),
            CBAM(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  #

            nn.Conv2d(256, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Softplus(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus(),
            # CBAM(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  #

            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Softplus(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus(),
            # CBAM(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  #

            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Softplus(),
            nn.Conv2d(32, 32, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus(),
            # CBAM(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 

            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Softplus(),
            nn.Conv2d(16, 16, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus(),
            # CBAM(16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 

            nn.Conv2d(16, 1, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Softplus(),
            nn.Conv2d(1, 1, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus()
            
        )
        self.init_weights()

    def init_weights(self):
        for m in self.upscale_layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        output = self.upscale_layers(x)


        return output





class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False)  
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x



