import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

from model.model import DCDiscriminator, DCGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN model")
    parser.add_argument('--save_dir', help='the dir to save logs and models')   
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--ngpus', type=int, default=1)
    args = parser.parse_args()
    return args



def normal_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def main():
    args = parse_args()
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    device = torch.device("cuda:0" if (
        torch.cuda.is_available() and args.ngpus > 0) else "cpu")

    # model
    Gconfig = dict(input_dim=100, output_dim=3, num_filters=[512, 256, 128, 64])

    # Discriminator
    Dconfig = dict(input_dim=3, output_dim=1, num_filters=[64, 128, 256, 512])

    img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder('/content/data', transform=img_transform)

    dataloader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=True,
                        num_workers=8)
    # train setting
    num_epochs = 10
    lr = 2e-4
    betas = (0.5, 0.999)
    batch_size = 128
    image_size = 64

    G = DCGenerator(Gconfig['input_dim'], Gconfig['num_filters'],
                    Gconfig['output_dim']).to(device)
    G.apply(normal_weights_init)
    D = DCDiscriminator(Dconfig['input_dim'], Dconfig['num_filters'],
                        Dconfig['output_dim']).to(device)
    D.apply(normal_weights_init)
    if (device.type == 'cuda') and (args.ngpus > 1):
        G = nn.DataParallel(G, list(range(args.ngpus)))
        D = nn.DataParallel(D, list(range(args.ngpus)))

    fixed_noise = torch.randn(64, Gconfig['input_dim'], 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # loss and optimizer
    criterion = nn.BCELoss()
    optimizerG = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizerD = optim.Adam(D.parameters(), lr=lr, betas=betas)

    # log
    img_list = []
    G_losses = []
    D_losses = []

    # Train
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            D.zero_grad()
            real = data[0].to(device)
            bs = real.size(0)

            # train D
            # Compute loss of true images, label is 1
            label = torch.full((bs, ), real_label, device=device)
            output = D(real).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Compute loss of fake images, label is 0
            noise = torch.randn(bs, Gconfig['input_dim'], 1, 1, device=device)
            fake = G(noise)
            label.fill_(fake_label)
            output = D(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()

            D_G_z1 = output.mean().item()
            errD = errD_fake + errD_real
            optimizerD.step()

            # train G
            # The purpose of the generator is to make the generated picture more realistic
            # label is 1
            G.zero_grad()
            label.fill_(real_label)
            output = D(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()

            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print(
                    '[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader), errD.item(),
                       errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            # iters += 1

        torch.save(G.state_dict(),
                   os.path.join(args.save_dir, "epoch_%d_G.pth" % epoch))
        torch.save(D.state_dict(),
                   os.path.join(args.save_dir, "epoch_%d_D.pth" % epoch))
        with torch.no_grad():
            fake = G(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    # plot_loss(G_losses, D_losses, args.save_dir)
    # plot_results(img_list, next(iter(dataloader))[0].to(device), args.save_dir)


if __name__ == '__main__':
    main()
