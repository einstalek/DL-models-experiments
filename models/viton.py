import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
import tqdm

from losses.perceptual_loss import PerceptualLoss


def sn_conv2d(in_ch, out_ch, ksize=1, padding=0, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize,
                                            stride=1, padding=padding,
                                            bias=bias))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=1, padding=0, bias=True):
        super(ConvBlock, self).__init__()
        self.conv = sn_conv2d(in_ch, out_ch, ksize, padding, bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RefinementModel(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=64, num_blocks=4):
        super(RefinementModel, self).__init__()
        self.blocks = []
        _in = in_dim
        for _ in range(num_blocks):
            self.blocks.append(ConvBlock(_in, hid_dim))
            _in = hid_dim
        self.blocks = nn.ModuleList(self.blocks)
        self.final = sn_conv2d(hid_dim, out_dim, 1, bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.final(x)
        return x


class RegressionModel(nn.Module):
    def __init__(self, grid_size, scale=0.1):
        super(RegressionModel, self).__init__()
        N = 2 * grid_size ** 2  # size of theta for TPS
        self.base = resnet18(num_classes=N)
        self.base.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7),
                                    stride=(2, 2), padding=(3, 3), bias=False)
        self.scale = scale

    def forward(self, x, y):
        inp = torch.cat([x, y], 1)
        return self.base(inp) * self.scale


def train_step(coarse_model, regr_model: RegressionModel, refine_net, grid_gen,
               desc, gt, raw_mask, deform_mask, raw_cloth,
               perceptual_loss: PerceptualLoss, crit,
               opt_coarse, opt_regr, opt_refine,
               mode='train'):
    if mode == 'train':
        opt_coarse.zero_grad()
        opt_regr.zero_grad()
        opt_refine.zero_grad()

    # Generate mask and coarse result from descriptor
    coarse_result = coarse_model(desc)
    coarse_mask, coarse_cloth = coarse_result[:, :1], coarse_result[:, 1:]

    coarse_loss = crit(deform_mask, coarse_mask) + crit(gt, coarse_cloth)
    coarse_loss += perceptual_loss(gt, coarse_cloth)

    # Generate Theta for TPS
    theta = regr_model(raw_mask, coarse_mask)

    deform_mask = deform_mask.repeat(1, 3, 1, 1)
    deform_cloth = torch.where(deform_mask > 0, gt, -torch.ones(1).cuda())
    raw_mask = raw_mask.repeat(1, 3, 1, 1)
    raw_cloth = torch.where(raw_mask > 0, raw_cloth, -torch.ones(1).cuda())

    # Apply TPS to mask and raw cloth
    grid = grid_gen(theta)
    warped_mask = F.grid_sample(raw_mask[:, :1], grid, padding_mode='border')
    warped_cloth = F.grid_sample(raw_cloth, grid, padding_mode='border')

    tps_loss = crit(warped_mask, deform_mask[:, :1]) + crit(warped_cloth, deform_cloth)
    tps_loss += perceptual_loss(deform_cloth, warped_cloth)

    coarse_inp = torch.cat([warped_cloth,
                            coarse_cloth], dim=1)
    # Generate refinement mask
    refined_mask = refine_net(coarse_inp).sigmoid()

    refined = refined_mask * warped_cloth + (1 - refined_mask) * coarse_cloth
    refine_loss = perceptual_loss(gt, refined, refinement=True)
    refine_loss += crit(gt, refined)
    refine_loss -= 0.1 * refined_mask.mean()

    if mode == 'train':
        coarse_loss.backward(retain_graph=True)
        tps_loss.backward(retain_graph=True)
        refine_loss.backward()
        opt_coarse.step()
        opt_regr.step()
        # Clip gradients for refinement net
        torch.nn.utils.clip_grad_norm_(refine_net.parameters(), max_norm=10.)
        opt_refine.step()
    return coarse_loss, tps_loss, refine_loss


def train_single_epoch(coarse_model, regr_model, refine_net, grid_gen,
                       perceptual_loss, crit, opt_coarse, opt_regr, opt_refine,
                       dataloader, logger, epoch, postfix_dict={}, device="cuda"):
    coarse_model.train()
    regr_model.train()
    refine_net.train()

    total_step = len(dataloader)
    total_loss = 0.
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, position=0)

    for i, (desc, gt, deform_mask, raw_mask) in tbar:
        desc = desc.to(device)
        gt = gt.to(device)
        deform_mask = deform_mask.to(device)
        raw_mask = raw_mask.to(device)
        raw_cloth = desc[:, 22:]

        coarse_loss, tps_loss, refine_loss = train_step(coarse_model, regr_model, refine_net, grid_gen,
                                                        desc, gt, raw_mask, deform_mask, raw_cloth,
                                                        perceptual_loss, crit,
                                                        opt_coarse, opt_regr, opt_refine)
        total_loss += coarse_loss.item() + tps_loss.item() + refine_loss.item()
        postfix_dict["train/closs"] = coarse_loss.item()
        postfix_dict["train/tloss"] = tps_loss.item()
        postfix_dict["train/rloss"] = refine_loss.item()
        f_epoch = epoch + i / total_step
        desc = '{:5s}'.format('train')
        desc += ', {:04d}/{:04d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)
    total_loss /= total_step
    logger.update(**{'train/loss': total_loss})


def eval_single_epoch(coarse_model, regr_model, refine_net, grid_gen,
                      perceptual_loss, crit, opt_coarse, opt_regr, opt_refine,
                      dataloader, logger, epoch, postfix_dict={}, device="cuda"):
    coarse_model.eval()
    regr_model.eval()
    refine_net.eval()
    total_step = len(dataloader)
    total_loss = 0.
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, position=0)

    for i, (desc, gt, deform_mask, raw_mask) in tbar:
        desc = desc.to(device)
        deform_mask = deform_mask.to(device)
        raw_mask = raw_mask.to(device)
        gt = gt.to(device)
        raw_cloth = desc[:, 22:]

        coarse_loss, tps_loss, refine_loss = train_step(coarse_model, regr_model, refine_net, grid_gen,
                                                        desc, gt, raw_mask, deform_mask, raw_cloth,
                                                        perceptual_loss, crit,
                                                        opt_coarse, opt_regr, opt_refine, mode='val')
        total_loss += coarse_loss.item() + tps_loss.item() + refine_loss.item()
        postfix_dict["val/closs"] = coarse_loss.item()
        postfix_dict["val/tloss"] = tps_loss.item()
        postfix_dict["val/rloss"] = refine_loss.item()
        f_epoch = epoch + i / total_step
        desc = '{:5s}'.format('val')
        desc += ', {:04d}/{:04d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)
    total_loss /= total_step
    logger.update(**{'val/loss': total_loss})


def train(coarse_model, regr_model, refine_net, grid_gen, opt_coarse, opt_regr, opt_refine,
          train_loader, val_loader,
          perceptual_loss, crit, logger, epochs=20):
    postfix_dict = {}
    for epoch in range(epochs):
        train_single_epoch(coarse_model, regr_model, refine_net, grid_gen,
                           perceptual_loss, crit, opt_coarse, opt_regr, opt_refine,
                           train_loader, logger, epoch, postfix_dict=postfix_dict)
        eval_single_epoch(coarse_model, regr_model, refine_net, grid_gen,
                          perceptual_loss, crit, opt_coarse, opt_regr, opt_refine,
                          val_loader, logger, epoch,
                          postfix_dict=postfix_dict)


def inference(coarse_model, regr_model, refine_net, grid_gen,
              desc, raw_mask):
    regr_model.eval()
    coarse_model.eval()
    refine_net.eval()

    raw_cloth = desc[:, 22:]
    coarse_result = coarse_model(desc)
    coarse_mask, coarse_cloth = coarse_result[:, :1], coarse_result[:, 1:]
    theta = regr_model(raw_mask, coarse_mask)

    raw_mask = raw_mask.repeat(1, 3, 1, 1)
    raw_cloth = torch.where(raw_mask > 0, raw_cloth, -torch.ones(1).cuda())

    grid = grid_gen(theta)
    warped_cloth = F.grid_sample(raw_cloth, grid, padding_mode='border')

    coarse_inp = torch.cat([warped_cloth,
                            coarse_cloth], dim=1)

    refined_mask = refine_net(coarse_inp).sigmoid()
    refined = refined_mask * warped_cloth + (1 - refined_mask) * coarse_cloth
    return refined
