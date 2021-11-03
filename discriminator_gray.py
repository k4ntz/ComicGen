import torch
from torch import nn
from torch.nn.utils import spectral_norm
from rational.torch import Rational
import utils


class Discriminator(nn.Module):
    def __init__(self, patch=True):
        super(Discriminator, self).__init__()
        self.patch = patch
        self.convs = nn.Sequential(
            spectral_norm(nn.Conv2d(1, 32, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(32, 32, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(32, 64, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(64, 64, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(128, 128, 3, 1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        if self.patch:
            self.layer_patch = nn.Conv2d(128, 1, 1, 1, 0)
        else:
            self.layer_patch = nn.Linear(128, 1)

        utils.init_weights(self)

    def forward(self, x_in):
        x = self.convs(x_in)
        if self.patch:
            x_out = self.layer_patch(x)
            return x_out
        else:
            x_out = self.layer_patch(torch.mean(x, dim=[2, 3]))
            return x_out


class Discriminator_Rational(nn.Module):
    def __init__(self, shared_pau=False, approx_func="leaky_relu", degrees=(5, 4), version="A"):
        super(Discriminator_Rational, self).__init__()
        if shared_pau:
            self.activation = Rational(approx_func=approx_func, degrees=degrees, version=version)

            self.convs = nn.Sequential(
                spectral_norm(nn.Conv2d(1, 32, 3, 2, 1)),
                self.activation,
                spectral_norm(nn.Conv2d(32,32,3,1,1)),
                self.activation,

                spectral_norm(nn.Conv2d(32,64,3,2,1)),
                self.activation,
                spectral_norm(nn.Conv2d(64,64,3,1,1)),
                self.activation,

                spectral_norm(nn.Conv2d(64,128,3,2,1)),
                self.activation,
                spectral_norm(nn.Conv2d(128,128,3,1,1)),
                self.activation,

                nn.Conv2d(128, 1, 1, 1, 0)
            )
        else:
            self.convs = nn.Sequential(
                spectral_norm(nn.Conv2d(1, 32, 3, 2, 1)),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                spectral_norm(nn.Conv2d(32,32,3,1,1)),
                Rational(approx_func=approx_func, degrees=degrees, version=version),

                spectral_norm(nn.Conv2d(32,64,3,2,1)),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                spectral_norm(nn.Conv2d(64,64,3,1,1)),
                Rational(approx_func=approx_func, degrees=degrees, version=version),

                spectral_norm(nn.Conv2d(64,128,3,2,1)),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                spectral_norm(nn.Conv2d(128,128,3,1,1)),
                Rational(approx_func=approx_func, degrees=degrees, version=version),

                nn.Conv2d(128, 1, 1, 1, 0)
            )

        # if self.patch:
        #     # self.layer_patch = spectral_norm(nn.Conv2d(128, 1, 1, 1, 0))
        #     self.layer_patch = nn.Conv2d(128, 1, 1, 1, 0)
        # else:
        #     self.layer_patch = nn.Linear(128, 1)

        utils.init_weights(self)

    def forward(self, x_in):
        x_out = self.convs(x_in)
        # if self.patch:
        #     x_out = self.layer_patch(x)
        # else:
        #     x_out = self.layer_patch(torch.mean(x, dim=[2, 3]))
        return x_out

# class Discriminator_Rational(nn.Module):
#     def __init__(self, approx_func="leaky_relu", degrees=(5, 4), version="A", patch=True):
#         super(Discriminator_Rational, self).__init__()
#         self.patch = patch
#
#         self.activation = Rational(approx_func=approx_func, degrees=degrees, version=version)
#
#         self.convs = nn.Sequential(
#             spectral_norm(nn.Conv2d(1, 32, 3, 2, 1)),
#             self.activation,
#             spectral_norm(nn.Conv2d(32,32,3,1,1)),
#             self.activation,
#
#             spectral_norm(nn.Conv2d(32,64,3,2,1)),
#             self.activation,
#             spectral_norm(nn.Conv2d(64,64,3,1,1)),
#             self.activation,
#
#             spectral_norm(nn.Conv2d(64,128,3,2,1)),
#             self.activation,
#             spectral_norm(nn.Conv2d(128,128,3,1,1)),
#             self.activation
#         )
#
#         if self.patch:
#             self.layer_patch = spectral_norm(nn.Conv2d(128, 1, 1, 1, 0))
#         else:
#             self.layer_patch = nn.Linear(128, 1)
#
#         utils.init_weights(self)
#
#     def forward(self, x_in):
#         x = self.convs(x_in)
#         if self.patch:
#             x_out = self.layer_patch(x)
#         else:
#             x_out = self.layer_patch(torch.mean(x, dim=[2, 3]))
#         return x_out


if __name__ == '__main__':
    disc = Discriminator()
    print(disc)
    # for para in disc.parameters():
    #     print(para)
    # w = torch.ones((32,3,3,3))
    # disc.spectral_norm(w)
    pass
