import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
import os
from tqdm import tqdm
from GAN.model import Discriminator, Generator
if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out


batch_size = 64
num_epoch = 1000
z_dimension = 49
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = ImageFolder('/content/data', transform=img_transform)

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=8)

D = Discriminator().cuda()  # discriminator model
G = Generator(z_dimension).cuda()  # generator model

criterion = nn.BCELoss()  # binary cross entropy

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

# train
for epoch in range(num_epoch):
    bar = tqdm(dataloader, dynamic_ncols=True)
    bar.set_description_str(f'{epoch}/{num_epoch}')
    i = 0
    for (img, _) in bar:
        i += 1
        num_img = img.size(0)
        # =================train discriminator
        real_img = Variable(img).cuda()
        real_label = Variable(torch.ones(num_img)).cuda()
        fake_label = Variable(torch.zeros(num_img)).cuda()

        # compute loss of real_img
        real_out = D(real_img)
        d_loss_real = criterion(real_out, real_label)
        real_scores = real_out  # closer to 1 means better

        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        fake_out = D(fake_img)
        d_loss_fake = criterion(fake_out, fake_label)
        fake_scores = fake_out  # closer to 0 means better

        # bp and optimize
        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ===============train generator
        # compute loss of fake_img
        z = Variable(torch.randn(num_img, z_dimension)).cuda()
        fake_img = G(z)
        output = D(fake_img)
        g_loss = criterion(output, real_label)

        # bp and optimize
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 2 == 0:
            bar.set_postfix_str('d_loss: {:.6f}, g_loss: {:.6f} '
                                'D real: {:.6f}, D fake: {:.6f}'.format(
                                    d_loss.item(), g_loss.item(),
                                    real_scores.data.mean(),
                                    fake_scores.data.mean()))
    if epoch == 0:
        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './dc_img/real_images.png')

    fake_images = to_img(fake_img.cpu().data)
    save_image(fake_images, './dc_img/fake_images-{}.png'.format(epoch + 1))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')
