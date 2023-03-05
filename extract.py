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
    rec_dir = os.path.join(result_dir, 'rec')
    mask_dir = os.path.join(result_dir, 'mask')

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

    video_path = os.path.join(os.path.abspath('.'), 'data', 'Video_%03d' % data_idx, 'Video_%03d.avi' % data_idx)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    # recVideo = cv2.VideoWriter(
    #     os.path.join(result_dir, 'Video_%03d_Rec.avi' % data_idx),
    #     cv2.VideoWriter_fourcc('I', '4', '2', '0'),
    #     fps,
    #     (x, y)
    # )

    for img_idx in range(nSample):
        img = imgs[img_idx]
        img_variable = Variable(torch.FloatTensor(img / 256)).unsqueeze(0).to(device)
        imgs_z_mu, imgs_z_logvar = vae.encoder(img_variable)
        imgs_z = vae.reparam(imgs_z_mu, imgs_z_logvar)
        imgs_rec = vae.decoder(imgs_z).cpu()
        imgs_rec = imgs_rec.data.numpy()
        img_i = imgs_rec[0].transpose(1, 0).reshape(x, y, 3)
        img_i = (img_i * 255).astype(np.uint8)
        cv2.imshow("rec", img_i)
        cv2.waitKey(10)
        # recVideo.write(img_i)
        # io.imsave(os.path.join(rec_dir, 'Img_%05d.bmp' % img_idx), img_i)
    # recVideo.release()
