"""
train.py
VAE模型训练,生成.pth存储到./models/下
"""

from __future__ import division

import os
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data
from VAE import VAE
from tensorboardX import SummaryWriter

from config.ModelConfig import VAEConfig
from config.BaseConfig import TRAIN_SET, IMAGE_SIZE, LATENT_DIM, BATCH, EPOCH, LR, FEATURE_ROW, FEATURE_COL

# logs
LogDir = os.path.join(os.path.abspath('.'), 'logs')

#
BETA = 0.80

# 损失函数
SparsityLoss = nn.L1Loss(reduction='sum')


def elbo_loss(recon_x, x, mu_z, logvar_z, img_size):
    l1_loss = SparsityLoss(recon_x, x.view(-1, 3, img_size))
    kld = -0.5 * BETA * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    return l1_loss + kld


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(' Processor is %s' % device)

    for data_idx in TRAIN_SET:
        print("---------------start data-set %d---------------" % data_idx)
        data_path = os.path.join(os.path.abspath('.'), 'data', 'Video_%03d' % data_idx, 'BMC2012_%03d.npy' % data_idx)
        model_path = os.path.join(os.path.abspath('.'), 'models', 'BMC2012_%03d.pth' % data_idx)

        loss_dir = os.path.join(LogDir, 'loss', 'BMC2012_%03d' % data_idx)
        if not os.path.exists(loss_dir):
            os.makedirs(loss_dir)

        loss_writer = SummaryWriter(loss_dir)

        imgs = np.load(data_path)
        imgs_tensor = torch.FloatTensor(imgs / 256)
        imgs_dataset = torch.utils.data.TensorDataset(imgs_tensor)
        train_loader = torch.utils.data.DataLoader(imgs_dataset, batch_size=BATCH, shuffle=True)

        vae = VAE(VAEConfig(
            device=device,
            img_size=IMAGE_SIZE[data_idx],
            latent_dim=LATENT_DIM[data_idx],
            feature_row=FEATURE_ROW[data_idx],
            feature_col=FEATURE_COL[data_idx]
        ))
        vae.to(device)
        vae_optimizer = optim.Adam(vae.parameters(), lr=LR)

        for epoch_idx in range(EPOCH):
            loss_vae_value = 0.00
            for batch_idx, data in enumerate(train_loader):
                data_vae = Variable(data[0]).to(device)
                vae_optimizer.zero_grad()
                recon_x, mu_z, logvar_z, z = vae.forward(data_vae)
                loss_vae = elbo_loss(recon_x, data_vae, mu_z, logvar_z, IMAGE_SIZE[data_idx])
                loss_vae.backward()
                loss_vae_value += loss_vae.item()
                vae_optimizer.step()
            loss = loss_vae_value / len(imgs_dataset)
            print('====> Epoch: %03d elbo_Loss : %0.8f' % ((epoch_idx + 1), loss))
            loss_writer.add_scalar('BMC2012_%03d ELBOLoss' % data_idx, loss, epoch_idx)
        torch.save(vae.state_dict(), model_path)
        loss_writer.close()


if __name__ == '__main__':
    main()
