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
            nn.Softplus(),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  #

            nn.Conv2d(256, 128, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Softplus(),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  #

            nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Softplus(),
            nn.Conv2d(64, 64, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  #

            nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Softplus(),
            nn.Conv2d(32, 32, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 

            nn.Conv2d(32, 16, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.Softplus(),
            nn.Conv2d(16, 16, kernel_size=1, padding=0, padding_mode='replicate'),
            nn.Softplus(),
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


