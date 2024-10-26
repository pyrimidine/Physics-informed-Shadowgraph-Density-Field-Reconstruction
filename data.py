import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.data_dir = data_dir
        self.image_files = sorted([os.path.join(data_dir, file) for file in os.listdir(data_dir) 
                                   if file.endswith(('png', 'jpg', 'jpeg', 'bmp'))])        
        
        
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        return image
    
    def ref(self, ref_file):
        path = os.path.join(self.data_dir, ref_file)
        image = Image.open(path).convert('L')
        image = self.transform(image)
        return image[:, None, :, :]
