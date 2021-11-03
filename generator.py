import torch.nn as nn
import utils
import torch.nn.functional as F
from rational.torch import Rational
import torch


def upsample(src, tgt):
    src = F.interpolate(src, size=tgt.shape[2:], mode='bilinear', align_corners=True)
    return src


class Generator(nn.Module):
    def __init__(self, use_enhance=False):
        super(Generator, self).__init__()
        self.use_enhance = use_enhance
        self._init_module()

    def _init_module(self):
        self.layer_3_32 = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_32_64 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_64_128 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.resblock_0 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, 3, 1, 1)
        )
        self.resblock_1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, 3, 1, 1)
        )
        self.resblock_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, 3, 1, 1)
        )
        self.resblock_3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, 3, 1, 1)
        )
        self.layer_128_64 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_64_32 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.layer_32_3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 3, 7, 1, 3)
        )
        # self.layer_128_64 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, 1, 1),
        #     nn.LeakyReLU(0.2, True)
        # )
        # self.layer_64_32 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, 1, 1),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(64, 64, 3, 1, 1),
        #     nn.LeakyReLU(0.2, True)
        # )
        # self.layer_32_3 = nn.Sequential(
        #     nn.Conv2d(128, 64, 3, 1, 1),
        #     nn.LeakyReLU(0.2, True),
        #     nn.Conv2d(64, 3, 7, 1, 3)
        # )

        utils.init_weights(self)

    def forward(self, x_in):
        x0 = self.layer_3_32(x_in)
        x1 = self.layer_32_64(x0)
        x2 = self.layer_64_128(x1)

        res0 = self.resblock_0(x2) + x2
        # res0 += x2
        res1 = self.resblock_1(res0) + res0
        # res1 += res0
        res2 = self.resblock_2(res1) + res1
        # res2 += res1
        res3 = self.resblock_3(res2) + res2
        # res3 += res2

        x3 = self.layer_128_64(res3)
        x3up = upsample(x3, x1)
        # x4 = self.layer_64_32(torch.cat(x3up, x1))
        x4 = self.layer_64_32(x3up+x1)
        x4up = upsample(x4, x0)
        # x_out = self.layer_32_3(torch.cat(x4up, x0))
        x_out = self.layer_32_3(x4up+x0)

        # h1, w1 = x3.shape[2], x3.shape[3]
        # x4 = F.interpolate(x3, (h1 * 2, w1 * 2), mode='bilinear', align_corners=True)
        # x5 = self.layer_64_32(x4 + x1)
        # h2, w2 = x5.shape[2], x5.shape[3]
        # x6 = F.interpolate(x5, (h2 * 2, w2 * 2), mode='bilinear', align_corners=True)
        # x7 = self.layer_32_3(x6 + x0)
        if self.use_enhance:
            x_out = torch.clamp(x_out, -1, 1)
        return x_out


class Generator_Rational(nn.Module):
    def __init__(self, shared_pau=False, approx_func="leaky_relu", degrees=(5, 4), version="A", use_enhance=False):
        super(Generator_Rational, self).__init__()
        self.use_enhance = use_enhance
        self.shared_pau = shared_pau
        self._init_module(approx_func, degrees, version)

    def _init_module(self, approx_func, degrees, version):
        if self.shared_pau:
            self.activation = Rational(approx_func=approx_func, degrees=degrees, version=version)
            self.layer_3_32 = nn.Sequential(
                nn.Conv2d(3, 32, 7, 1, 3),
                self.activation
            )
            self.layer_32_64 = nn.Sequential(
                nn.Conv2d(32, 32, 3, 2, 1),
                self.activation,
                nn.Conv2d(32, 64, 3, 1, 1),
                self.activation
            )
            self.layer_64_128 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 2, 1),
                self.activation,
                nn.Conv2d(64, 128, 3, 1, 1),
                self.activation
            )
            self.resblock_0 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                self.activation,
                nn.Conv2d(128, 128, 3, 1, 1)
            )
            self.resblock_1 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                self.activation,
                nn.Conv2d(128, 128, 3, 1, 1)
            )
            self.resblock_2 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                self.activation,
                nn.Conv2d(128, 128, 3, 1, 1)
            )
            self.resblock_3 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                self.activation,
                nn.Conv2d(128, 128, 3, 1, 1)
            )
            self.layer_128_64 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),
                self.activation
            )
            self.layer_64_32 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                self.activation,
                nn.Conv2d(64, 32, 3, 1, 1),
                self.activation
            )
            self.layer_32_3 = nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                self.activation,
                nn.Conv2d(32, 3, 7, 1, 3)
            )
        else:
            self.layer_3_32 = nn.Sequential(
                nn.Conv2d(3, 32, 7, 1, 3),
                Rational(approx_func=approx_func, degrees=degrees, version=version)
            )
            self.layer_32_64 = nn.Sequential(
                nn.Conv2d(32, 32, 3, 2, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                nn.Conv2d(32, 64, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version)
            )
            self.layer_64_128 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 2, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                nn.Conv2d(64, 128, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version)
            )
            self.resblock_0 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                nn.Conv2d(128, 128, 3, 1, 1)
            )
            self.resblock_1 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                nn.Conv2d(128, 128, 3, 1, 1)
            )
            self.resblock_2 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                nn.Conv2d(128, 128, 3, 1, 1)
            )
            self.resblock_3 = nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                nn.Conv2d(128, 128, 3, 1, 1)
            )
            self.layer_128_64 = nn.Sequential(
                nn.Conv2d(128, 64, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version)
            )
            self.layer_64_32 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                nn.Conv2d(64, 32, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version)
            )
            self.layer_32_3 = nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1),
                Rational(approx_func=approx_func, degrees=degrees, version=version),
                nn.Conv2d(32, 3, 7, 1, 3)
            )

        utils.init_weights(self)

    def forward(self, x_in):
        x0 = self.layer_3_32(x_in)
        x1 = self.layer_32_64(x0)
        x2 = self.layer_64_128(x1)

        res0 = self.resblock_0(x2) + x2
        # res0 += x2
        res1 = self.resblock_1(res0) + res0
        # res1 += res0
        res2 = self.resblock_2(res1) + res1
        # res2 += res1
        res3 = self.resblock_3(res2) + res2
        # res3 += res2

        x3 = self.layer_128_64(res3)
        x3up = upsample(x3, x1)
        x4 = self.layer_64_32(x3up + x1)
        x4up = upsample(x4, x0)
        x_out = self.layer_32_3(x4up + x0)

        # h1, w1 = x3.shape[2], x3.shape[3]
        # x4 = F.interpolate(x3, (h1 * 2, w1 * 2), mode='bilinear', align_corners=True)
        # x5 = self.layer_64_32(x4 + x1)
        # h2, w2 = x5.shape[2], x5.shape[3]
        # x6 = F.interpolate(x5, (h2 * 2, w2 * 2), mode='bilinear', align_corners=True)
        # x7 = self.layer_32_3(x6 + x0)
        if self.use_enhance:
            x_out = torch.clamp(x_out, -1, 1)
        return x_out


# class Generator_Rational(nn.Module):
#     def __init__(self, approx_func="leaky_relu", degrees=(5, 4), version="A", use_enhance=False):
#         super(Generator_Rational, self).__init__()
#         self.use_enhance = use_enhance
#
#         self._init_module(approx_func, degrees, version)
#
#     def _init_module(self, approx_func, degrees, version):
#         self.activation = Rational(approx_func=approx_func, degrees=degrees, version=version)
#         self.layer_3_32 = nn.Sequential(
#             nn.Conv2d(3, 32, 7, 1, 3),
#             self.activation
#         )
#         self.layer_32_64 = nn.Sequential(
#             nn.Conv2d(32, 32, 3, 2, 1),
#             self.activation,
#             nn.Conv2d(32, 64, 3, 1, 1),
#             self.activation
#         )
#         self.layer_64_128 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 2, 1),
#             self.activation,
#             nn.Conv2d(64, 128, 3, 1, 1),
#             self.activation
#         )
#         self.resblock_0 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1),
#             self.activation,
#             nn.Conv2d(128, 128, 3, 1, 1)
#         )
#         self.resblock_1 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1),
#             self.activation,
#             nn.Conv2d(128, 128, 3, 1, 1)
#         )
#         self.resblock_2 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1),
#             self.activation,
#             nn.Conv2d(128, 128, 3, 1, 1)
#         )
#         self.resblock_3 = nn.Sequential(
#             nn.Conv2d(128, 128, 3, 1, 1),
#             self.activation,
#             nn.Conv2d(128, 128, 3, 1, 1)
#         )
#         self.layer_128_64 = nn.Sequential(
#             nn.Conv2d(128, 64, 3, 1, 1),
#             self.activation
#         )
#         self.layer_64_32 = nn.Sequential(
#             nn.Conv2d(64, 64, 3, 1, 1),
#             self.activation,
#             nn.Conv2d(64, 32, 3, 1, 1),
#             self.activation
#         )
#         self.layer_32_3 = nn.Sequential(
#             nn.Conv2d(32, 32, 3, 1, 1),
#             self.activation,
#             nn.Conv2d(32, 3, 7, 1, 3)
#         )
#
#         utils.init_weights(self)
#
#     def forward(self, x_in):
#         x0 = self.layer_3_32(x_in)
#         x1 = self.layer_32_64(x0)
#         x2 = self.layer_64_128(x1)
#
#         res0 = self.resblock_0(x2) + x2
#         # res0 += x2
#         res1 = self.resblock_1(res0) + res0
#         # res1 += res0
#         res2 = self.resblock_2(res1) + res1
#         # res2 += res1
#         res3 = self.resblock_3(res2) + res2
#         # res3 += res2
#
#         x3 = self.layer_128_64(res3)
#         x3up = upsample(x3, x1)
#         x4 = self.layer_64_32(x3up + x1)
#         x4up = upsample(x4, x0)
#         x_out = self.layer_32_3(x4up + x0)
#
#         # h1, w1 = x3.shape[2], x3.shape[3]
#         # x4 = F.interpolate(x3, (h1 * 2, w1 * 2), mode='bilinear', align_corners=True)
#         # x5 = self.layer_64_32(x4 + x1)
#         # h2, w2 = x5.shape[2], x5.shape[3]
#         # x6 = F.interpolate(x5, (h2 * 2, w2 * 2), mode='bilinear', align_corners=True)
#         # x7 = self.layer_32_3(x6 + x0)
#         if self.use_enhance:
#             x_out = torch.clamp(x_out, -1, 1)
#         return x_out


if __name__ == "__main__":
    gene = Generator()
    print(gene)
    # for m in list(gene.modules())[2:]:
    #     print(m)
    #     print(m.weight)

