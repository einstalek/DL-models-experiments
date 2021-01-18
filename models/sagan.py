import torch
import torch.nn as nn
import tqdm
from IPython import display
import matplotlib.pyplot as plt


def sn_conv2d(in_ch, out_ch, ksize=1, strides=1, padding=0, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize,
                                            stride=strides, padding=padding,
                                            bias=bias))

class SelfAttention(nn.Module):
    def __init__(self, in_ch, scale=8):
        super(SelfAttention, self).__init__()
        self.scale = scale
        self.Q = sn_conv2d(in_ch, in_ch//scale)
        self.K = sn_conv2d(in_ch, in_ch//scale)
        self.V = sn_conv2d(in_ch, in_ch//scale)
        self.out = sn_conv2d(in_ch//scale, in_ch)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bsize, in_ch, h, w = x.size()
        queries = self.Q(x).view(-1, in_ch // self.scale, h*w)
        keys = self.K(x).view(-1, h*w, in_ch // self.scale)
        attn = torch.bmm(queries, keys).softmax(-1)  # (bsize, N, N)
        values = self.V(x).view(-1, in_ch // self.scale, h*w)
        out = torch.bmm(attn, values).view(-1, in_ch // self.scale, h, w)
        out = self.out(out)
        return out * self.gamma + x

class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GenBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = sn_conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, lat_dim, conv_dim):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim
        self.linear = nn.Linear(lat_dim, conv_dim * 16 * 4 * 4)
        self.block1 = GenBlock(conv_dim * 16, conv_dim * 16)
        self.block2 = GenBlock(conv_dim * 16, conv_dim * 8)
        self.sa = SelfAttention(conv_dim * 8)
        self.block3 = GenBlock(conv_dim * 8, conv_dim * 4)
        self.block4 = GenBlock(conv_dim * 4, conv_dim * 4)
        self.block5 = GenBlock(conv_dim * 4, conv_dim * 2)
        self.block6 = GenBlock(conv_dim * 2, conv_dim)
        self.final = sn_conv2d(conv_dim, 3, 3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x).view(-1, self.conv_dim * 16, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.sa(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.final(x)
        return self.tanh(x)


class Discriminator(nn.Module):
    def __init__(self, nc=3,  ndf=64):
        super(Discriminator, self).__init__()
        self.main = [
            sn_conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True)]
        in_ch = ndf
        for _ in range(5):
            self.main.extend([
                sn_conv2d(in_ch, in_ch * 2, 4, strides=2, padding=1, bias=False),
                nn.BatchNorm2d(in_ch * 2),
                nn.LeakyReLU(0.1, inplace=True),
            ])
            in_ch *= 2
        self.main.extend([
            sn_conv2d(in_ch, 1, 4, strides=1, padding=0, bias=False),
            nn.Sigmoid()
        ])
        self.main = nn.Sequential(*self.main)

    def forward(self, input):
        return self.main(input)


def train_step(real_images, gen, disc, optimizer_g, optimizer_d, e_crit,
               device='cuda'):
    bsize = real_images.size(0)
    noise = torch.randn(bsize, 64).to(device)
    fake_images = gen(noise)

    real_label_e = torch.full((bsize, 1, 1, 1), 1, dtype=torch.float, device=device)
    fake_label_e = torch.full((bsize, 1, 1, 1), 0, dtype=torch.float, device=device)

    disc.zero_grad()
    disc_real_e = disc(real_images)
    disc_fake_e = disc(fake_images)

    # disc loss
    disc_loss = e_crit(disc_real_e, real_label_e) + e_crit(disc_fake_e, fake_label_e)
    disc_loss.backward()
    optimizer_d.step()

    # gen loss
    gen.zero_grad()
    for _ in range(1):
        noise = torch.randn(bsize, 64).to(device)
        fake_images = gen(noise)
        disc_fake_e = disc(fake_images)
        gen_loss = e_crit(disc_fake_e, real_label_e)
        gen_loss.backward()
        optimizer_g.step()

    return disc_loss, gen_loss


def train_single_epoch(gen, disc, optimizer_g, optimizer_d, dataloader, logger,
                       e_crit, epoch, postfix_dict={}, device="cuda"):
    gen.train()
    disc.train()
    total_step = len(dataloader)
    total_loss = {"gloss": 0., "dloss": 0.}
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, position=0)
    for i, images in tbar:
        images = images.to(device)
        disc_loss, gen_loss = train_step(images, gen, disc, optimizer_g, optimizer_d,
                                         e_crit)
        total_loss["dloss"] += disc_loss.item()
        total_loss["gloss"] += gen_loss.item()
        postfix_dict['dloss'] = disc_loss.item()
        postfix_dict['gloss'] = gen_loss.item()
        f_epoch = epoch + i / total_step
        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)
    total_loss["gloss"] /= total_step
    total_loss["dloss"] /= total_step
    logger.update(**total_loss)


def train(gen, disc, optimizer_g, optimizer_d, dataloader, logger, epochs=10, clip_grad_norm=None):
    e_crit = nn.BCELoss()
    for epoch in range(epochs):
        train_single_epoch(gen, disc, optimizer_g, optimizer_d, dataloader, logger,
                           e_crit, epoch, device="cuda")
        noise = torch.randn(1, 64).to("cuda")
        gen.eval()
        x = 0.5 * gen(noise) + 0.5
        display.clear_output(wait=True)
        plt.imshow(x.detach().cpu().numpy()[0].transpose(1, 2, 0))
        plt.show()


