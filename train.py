import os
import argparse
from datetime import datetime

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from models import *
from data import *
from loss import *

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=48, help='Batch size for raw images')
parser.add_argument('--imgSize', type=tuple, default=(928, 1152)) # (h, w)
parser.add_argument('--work_path', type=str, help='Working folder')
parser.add_argument('--nEpochs', type=int, default=100, help='Number of epochs for training')
parser.add_argument('--LR', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--cuda', type=bool, default=True, help='Enables CUDA')
parser.add_argument('--train_pinn_only', type=bool, default=True, help='Train only the PINN model')
parser.add_argument('--save_intv', type=int, default=2, help='Model save interval')

opt = parser.parse_args(args=[])

dataset_path = os.path.join(opt.work_path, './dataset')

try:
    os.makedirs(os.path.join(opt.work_path, './result'))
    os.makedirs(os.path.join(opt.work_path, './model'))
except OSError:
    pass

output_path = os.path.join(opt.work_path, './result')
model_path = os.path.join(opt.work_path, './model')

dataset = ImageDataset(dataset_path, opt.imgSize)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=4)

model_1 = ResNetCNN_AE()
criterion_1 = nn.MSELoss()
optimizer_1 = optim.Adam(model_1.parameters(), lr=opt.LR * 1e-2)
scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=10, gamma=0.5)

model_2 = ImprovedDecoder()
criterion_2 = nn.SmoothL1Loss()
criterion_3 = nn.MSELoss()
optimizer_2 = optim.Adam(model_2.parameters(), lr=opt.LR)
scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=10, gamma=0.8)

TIMESTAMP = "{0:%m-%d_%H.%M/}".format(datetime.now())

if __name__ == '__main__':
    print('Training started')

    ref_img = dataset.ref('./Image_ref.bmp')

    if torch.cuda.device_count() > 1:
        model_1 = nn.DataParallel(model_1, range(torch.cuda.device_count()))
        model_2 = nn.DataParallel(model_2, range(torch.cuda.device_count()))
        print('Multi-GPU training enabled:', torch.cuda.device_count())

    for epoch in range(opt.nEpochs):
        for i, raw_img in enumerate(dataloader):
            if opt.cuda:
                raw_img = raw_img.cuda()
                ref_img = ref_img.cuda()
                norm_img = torch.sigmoid((raw_img - ref_img) / torch.mean(ref_img))
                model_1.cuda()
                model_2.cuda()

            if opt.train_pinn_only:
                with torch.no_grad():
                    model_1.eval()
                    feature, fake_img = model_1(norm_img)
                    loss_ae = torch.tensor(0.0)
            else:
                feature, fake_img = model_1(norm_img)
                recon_item = criterion_1(norm_img, fake_img)
                loss_ae = recon_item

                model_1.zero_grad()
                loss_ae.backward()
                optimizer_1.step()

            scale_log_RI = model_2(feature.detach())
            with torch.no_grad():
                model_1.eval()
                feature_ref, _ = model_1(0.5 * torch.ones_like(ref_img))

            bg_scale_log_RI = model_2(feature_ref.detach())
            L_res, R_res = possion_eq(raw_img, scale_log_RI, ref_img, X_len=88.47*1e-3, Y_len=109.83*1e-3, L=1.3+0.78, D=0.0015)
            PDE_res = torch.abs(torch.mean(L_res - R_res))

            loss_eq = criterion_2(L_res, R_res)
            loss_bg = criterion_3(bg_scale_log_RI, 0.27 * torch.ones_like(bg_scale_log_RI)) 
            # 0.27 is the normalized refractive index of air: 1e3 * log(IR_air) 

            loss_pinn = 1.0 * loss_eq + 10 * loss_bg

            model_2.zero_grad()
            loss_pinn.backward()
            torch.nn.utils.clip_grad_norm_(model_2.parameters(), max_norm=1.0)
            optimizer_2.step()

            scale_log_RI = 1e3 * (torch.exp(scale_log_RI * 1e-3) - 1)

            if i % 10 == 0:
                scheduler_2.step()
                scale_log_RI_img = scale_log_RI[0].detach().cpu().squeeze(0).numpy()
                print(f'Epoch [{epoch + 1}/{opt.nEpochs}], Step [{i + 1}/{len(dataloader)}], Loss_pinn: {loss_pinn.item():.3e}, Loss_ae: {loss_ae.item():.3e}, Residual_pde: {PDE_res.item():.3e}')

                output_image_norm = cv2.equalizeHist(cv2.normalize(scale_log_RI_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
                plt.imsave(os.path.join(output_path, f'output_{epoch + 1}.png'), output_image_norm, cmap='gray')

                plt.clf()
                plt.imshow(scale_log_RI_img, cmap='jet')
                plt.colorbar(label='Value')
                plt.title('1e3 * log(n)')
                plt.savefig(os.path.join(output_path, f'{epoch + 1}_RI.png'))

        if (epoch + 1) % opt.save_intv == 0:
            print('Saving model...')
            torch.save(model_1.module.state_dict(), os.path.join(model_path, './ae.pkl'))
            torch.save(model_2.module.state_dict(), os.path.join(model_path, './de.pkl'))
            torch.cuda.empty_cache()
