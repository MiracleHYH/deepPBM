from __future__ import division

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data
from VAE import VAE
from config.ModelConfig import VAEConfig
import os

# 训练参数
BATCH = 140
EPOCH = 200
LR = 1e-4

#
BETA = 0.80

# 数据集参数
IMAGE_SIZE = [0, 320 * 240, 352 * 288, 320 * 240, 320 * 240, 320 * 240, 320 * 240, 320 * 240, 320 * 240, 352 * 288]
LATENT_DIM = [0, 30, 30, 20, 35, 30, 30, 30, 1, 20]

# 损失函数
SparsityLoss = nn.L1Loss(reduction='sum')


def elbo_loss(recon_x, x, mu_z, logvar_z, img_size):
    l1_loss = SparsityLoss(recon_x, x.view(-1, 3, img_size))
    kld = -0.5 * BETA * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
    return l1_loss + kld


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(' Processor is %s' % device)

    for data_idx in range(1, 10):
        print("---------------start data-set %d---------------" % data_idx)
        data_path = os.path.join(os.path.abspath('.'), 'data', 'Video_%03d' % data_idx, 'BMC2012_%03d.npy' % data_idx)
        model_path = os.path.join(os.path.abspath('.'), 'models', 'Video_%03d' % data_idx, 'BMC2012_%03d.pth' % data_idx)

        imgs = np.load(data_path)
        imgs_tensor = torch.FloatTensor(imgs / 256)
        imgs_dataset = torch.utils.data.TensorDataset(imgs_tensor)
        train_loader = torch.utils.data.DataLoader(imgs_dataset, batch_size=BATCH, shuffle=True)

        vae = VAE(VAEConfig(device=device, img_size=IMAGE_SIZE[data_idx], latent_dim=LATENT_DIM[data_idx]))
        vae.to(device)
        vae_optimizer = optim.Adam(vae.parameters(), lr=LR)

        for epoch_idx in range(EPOCH):
            loss_vae_value = 0.00
            for batch_idx, data in enumerate(train_loader):
                data_vae = Variable(data).to(device)
                vae_optimizer.zero_grad()
                recon_x, mu_z, logvar_z, z = vae.forward(data_vae)
                loss_vae = elbo_loss(recon_x, data_vae, mu_z, logvar_z, IMAGE_SIZE[data_idx])
                loss_vae.backward()
                loss_vae_value += loss_vae.item()
                vae_optimizer.step()
            print('====> Epoch: %d elbo_Loss : %0.8f' % ((epoch_idx + 1), loss_vae_value / len(imgs_dataset)))
        torch.save(vae.state_dict(), model_path)


if __name__ == '__main__':
    main()
