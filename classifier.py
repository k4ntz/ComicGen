import torch
from torch import nn
from torch.nn.utils import spectral_norm
# from rational.torch import Rational
import utils


class Discriminator(nn.Module):
    def __init__(self, c_in, channel=32):
        super(Discriminator, self).__init__()

        self.convs = nn.Sequential(
            spectral_norm(nn.Conv2d(c_in, channel, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(channel, channel, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(channel, channel * 2, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(channel * 2, channel * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),

            spectral_norm(nn.Conv2d(channel * 2, channel * 4, 3, 2, 1)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Conv2d(channel * 4, channel * 4, 3, 1, 1)),
            nn.LeakyReLU(0.2, True),

            nn.Linear(128, 1)
        )

        utils.init_weights(self)

    def forward(self, x_in):
        x_out = self.convs(x_in)
        return x_out


if __name__ == '__main__':
    face_photo_dir = 'dataset/photo_face'
    face_photo_list = utils.load_image_list(face_photo_dir)
    scenery_photo_dir = 'dataset/photo_scenery'
    scenery_photo_list = utils.load_image_list(scenery_photo_dir)

    face_cartoon_dir = 'dataset/cartoon_face'
    face_cartoon_list = utils.load_image_list(face_cartoon_dir)
    scenery_cartoon_dir = 'dataset/cartoon_scenery'
    scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir)

    classifier_1 = Discriminator()


