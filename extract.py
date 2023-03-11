"""
extract.py
前景提取
读取models/下模型文件
读取data/Video_{001,002,003...}/img/下待处理的图片，
生成重建图像存入results/Video_{001,002,003...}/rec/下
提取前景mask存入results/Video_{001,002,003...}/mask/下
"""

from __future__ import division

import os
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from VAE import VAE
from config.ModelConfig import VAEConfig
from config.BaseConfig import TRAIN_SET, IMAGE_SIZE, LATENT_DIM, BATCH, EPOCH, LR, FEATURE_ROW, FEATURE_COL
from skimage import io

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(' Processor is %s' % device)

for data_idx in TRAIN_SET:
    data_path = os.path.join(os.path.abspath('.'), 'data', 'Video_%03d' % data_idx, 'BMC2012_%03d.npy' % data_idx)
    model_path = os.path.join(os.path.abspath('.'), 'models', 'BMC2012_%03d.pth' % data_idx)
    result_dir = os.path.join(os.path.abspath('.'), 'results', 'Video_%03d' % data_idx)
    real_dir = os.path.join(result_dir, 'real')
    rec_dir = os.path.join(result_dir, 'rec')
    mask_dir = os.path.join(result_dir, 'mask')

    if not os.path.exists(real_dir):
        os.makedirs(real_dir)
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)
    if not os.path.exists(mask_dir):
        os.makedirs(mask_dir)

    vae = VAE(VAEConfig(
        device=device,
        img_size=IMAGE_SIZE[data_idx],
        latent_dim=LATENT_DIM[data_idx],
        feature_row=FEATURE_ROW[data_idx],
        feature_col=FEATURE_COL[data_idx]
    ))
    vae.load_state_dict(torch.load(model_path))
    vae.to(device)
    imgs = np.load(data_path)
    nSample, ch, x, y = imgs.shape

    for img_idx in range(nSample):
        img_o = imgs[img_idx].transpose(1, 2, 0).astype(np.uint8)

        img_variable = Variable(torch.FloatTensor(imgs[img_idx] / 256)).unsqueeze(0).to(device)

        imgs_z_mu, imgs_z_logvar = vae.encoder(img_variable)
        imgs_z = vae.reparam(imgs_z_mu, imgs_z_logvar)
        imgs_rec = vae.decoder(imgs_z).cpu()

        img_i = imgs_rec.data.numpy()[0].transpose(1, 0).reshape(x, y, 3)
        img_i = (img_i * 255).astype(np.uint8)

        sub_img = cv2.absdiff(img_o, img_i)
        sub_img = cv2.GaussianBlur(sub_img, (3, 3), 0, 0)
        gray_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        th, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_TRIANGLE)
        res_img = cv2.erode(gray_img, np.ones((3, 3), np.uint8))
        # cv2.imshow("real & recover", np.hstack([img_o, img_i]))
        # cv2.imshow("mask", res_img)
        # if cv2.waitKey(30) & 0xFF == 27:
        #     break
        io.imsave(os.path.join(real_dir, 'Img_%05d.bmp' % img_idx), img_o)
        io.imsave(os.path.join(rec_dir, 'Img_%05d.bmp' % img_idx), img_i)
        io.imsave(os.path.join(mask_dir, 'Img_%05d.bmp' % img_idx), res_img)
