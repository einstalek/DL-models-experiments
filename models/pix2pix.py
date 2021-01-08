import os

import tqdm
import torch
import torch.nn as nn
from torch import optim


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            *discriminator_block(512,  1024),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(1024, 1, 4, padding=1, bias=False),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


def train_single_epoch(gen, disc, data_loader, b_size, epoch,
                       l1_criterion, e_criterion,
                       optimizer_gen, optimizer_disc, lam=20.):
    gen.train()
    disc.train()
    total_steps = len(data_loader)
    postfix_dict = {}
    tbar = tqdm.tqdm(enumerate(data_loader), total=total_steps, position=0, leave=True)

    for i, (img_a, img_b) in tbar:
        img_a = img_a.cuda()
        img_b = img_b.cuda()

        gen.zero_grad()
        disc.zero_grad()

        fake_b = gen(img_a)
        label = torch.full((b_size, 1, 1, 1), 0, dtype=torch.float, device='cuda')
        disc_fake = disc(img_a, fake_b)
        loss_disc_fake = e_criterion(disc_fake, label)
        loss_disc_fake.backward(retain_graph=True)

        label.fill_(1)
        disc_real = disc(img_a, img_b)
        loss_disc_real = e_criterion(disc_real, label)
        loss_disc_real.backward()
        postfix_dict["train/disc_loss"] = 0.5 * (loss_disc_fake + loss_disc_real).item()

        gen_loss = e_criterion(disc_fake, label) + lam * l1_criterion(img_b, fake_b)
        gen_loss.backward()
        postfix_dict["train/gen_loss"] = gen_loss.item()

        optimizer_gen.step()
        optimizer_disc.step()

        f_epoch = epoch + i / total_steps
        desc = '{:04d}/{:04d}, {:.2f} epoch'.format(i, total_steps, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)


def save_model(model, epoch, dir, name):
    fp = os.path.join(dir, name + '.pth')
    ckpt = {'state_dict': model.state_dict(),
            'epoch': epoch}
    torch.save(ckpt, fp)


def train(gen, disc, data_loader, epochs, dir, b_size=16, lr=5e-5, device="cuda"):
    gen = gen.to(device)
    disc = disc.to(device)
    e_criterion = nn.BCELoss()
    l1_criterion = nn.L1Loss()

    optimizer_gen = optim.Adam(gen.parameters(), lr=lr)
    optimizer_disc = optim.Adam(disc.parameters(), lr=lr)

    for epoch in range(epochs):
        train_single_epoch(gen, disc, data_loader, b_size, epoch,
                           l1_criterion, e_criterion,
                           optimizer_gen, optimizer_disc)
        save_model(gen, epoch, dir, "gen")
        save_model(disc, epoch, dir, "disc")

