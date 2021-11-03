import numpy as np
import torch
import torch.nn.functional as F
from arguments import args
from torch import nn
from collections import OrderedDict


VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19(nn.Module):
    def __init__(self, vgg19_npy_path=None):
        super(Vgg19, self).__init__()
        self.config = [64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 256, 'MP', 512, 512, 512, 512, 'MP']
        self.layers = self.construct_layers(self.config)

        if vgg19_npy_path:
            self.load_state_dict(self.load_model(vgg19_npy_path))
            print('VGG19 model loaded')

    def load_model(self, vgg19_npy_path):
        np_vgg = np.load(vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
        mapping = {'conv1_1': 'layers.0',
                   'conv1_2': 'layers.2',
                   'conv2_1': 'layers.5',
                   'conv2_2': 'layers.7',
                   'conv3_1': 'layers.10',
                   'conv3_2': 'layers.12',
                   'conv3_3': 'layers.14',
                   'conv3_4': 'layers.16',
                   'conv4_1': 'layers.19',
                   'conv4_2': 'layers.21',
                   'conv4_3': 'layers.23',
                   'conv4_4': 'layers.25'}
        data_dict = {}
        for k, [w, b] in np_vgg.items():
            if k in ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                     'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4']:
                data_dict[mapping[k] + '.weight'] = torch.as_tensor(np.transpose(w, (3, 2, 0, 1)))
                data_dict[mapping[k] + '.bias'] = torch.as_tensor(b)
        return data_dict

    def construct_layers(self, config):
        layers = []
        in_channels = 3
        for v in config:
            if v == 'MP':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, rgb):
        rgb_scaled = (rgb+1)*127.5
        b, g, r = torch.split(rgb_scaled, 1, dim=1)
        x = torch.cat((b-VGG_MEAN[0], g-VGG_MEAN[1], r-VGG_MEAN[2]), dim=1)

        module_list = list(self.layers.modules())
        for l in module_list[1:27]:  # conv4_4
            x = l(x)
        return x


if __name__ == '__main__':
    pass
    import torchvision
    # vgg = torchvision.models.vgg19(pretrained=True)
    # print(vgg)
    vgg=Vgg19()
    print(vgg)

    a = torch.randn((16,3,256,256))
    b = vgg(a)
