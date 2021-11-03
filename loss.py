import torch
import torch.nn.functional as F


def gan_loss_d(real, fake):
    return torch.mean(F.logsigmoid(real)+F.logsigmoid(1 - fake))


def gan_loss_g(fake):
    return torch.mean(F.logsigmoid(fake))


def lsgan_loss_d(real, fake):
    return 0.5 * (torch.mean((real - 1) ** 2) + torch.mean(fake ** 2))


def lsgan_loss_g(fake):
    return torch.mean((fake - 1) ** 2)


def total_variation_loss(image, k_size=1):
    _, c, h, w = image.shape
    tv_h = torch.mean((image[:, :, k_size:, :] - image[:, :, :h - k_size, :])**2)
    tv_w = torch.mean((image[:, :, :, k_size:] - image[:, :, :, :w - k_size]) ** 2)
    tv_loss = (tv_h + tv_w)/(c*h*w)
    return tv_loss


if __name__ == '__main__':
    # x = torch.randn((1, 3, 5, 5))
    pass