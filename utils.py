import torch
from torch import nn
from joblib import Parallel, delayed
import os
from PIL import Image
import numpy as np
from skimage import segmentation, color
from selective_search.util import switch_color_space
from selective_search.structure import HierarchicalGrouping
# from arguments import args
from torchvision import transforms
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def color_shift(image, mode='uniform'):
    b, g, r = torch.split(image, 1, dim=1)
    b_weight = 1
    g_weight = 1
    r_weight = 1
    if mode == 'normal':
        b_weight = nn.init.normal_(torch.Tensor(1,).to(device), mean=0.114, std=0.1)
        g_weight = nn.init.normal_(torch.Tensor(1,).to(device), mean=0.587, std=0.1)
        r_weight = nn.init.normal_(torch.Tensor(1,).to(device), mean=0.299, std=0.1)
    elif mode == 'uniform':
        b_weight = nn.init.uniform_(torch.Tensor(1,).to(device), a=0.014, b=0.214)
        g_weight = nn.init.uniform_(torch.Tensor(1,).to(device), a=0.487, b=0.687)
        r_weight = nn.init.uniform_(torch.Tensor(1,).to(device), a=0.199, b=0.399)
    output = (b_weight * b + g_weight * g + r_weight * r) / (b_weight + g_weight + r_weight)
    output = output.float()
    return output


def load_image_list(data_dir):
    name_list = list()
    for name in os.listdir(data_dir):
        name_list.append(os.path.join(data_dir, name))
    name_list.sort()
    return name_list


def next_batch(filename_list, batch_size):
    idx = np.arange(0, len(filename_list))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = []
    for i in range(batch_size):
        image = cv2.imread(filename_list[idx[i]])
        # print(image.shape)
        # image = resize_crop(image)
        # if image is None:
        #     import ipdb; ipdb.set_trace()
        image = image.astype(np.float32) / 127.5 - 1
        # image = image.astype(np.float32)/255
        image = np.transpose(image, (2, 0, 1))
        batch_data.append(image)
    batch_data = np.array(batch_data)
    # print(batch_data.shape)
    return torch.from_numpy(batch_data).to(device)


# def next_batch(filename_list, batch_size):
#     idx = np.arange(0, len(filename_list))
#     np.random.shuffle(idx)
#     idx = idx[:batch_size]
#     batch_data = []
#
#     transformer = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#     ])
#     for i in range(batch_size):
#         image = Image.open(filename_list[idx[i]])
#         image = transformer(image)
#         image = image.unsqueeze(0)
#         batch_data.append(image)
#     return torch.cat(batch_data, 0).to(device)


def init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def label2rgb(label_field, image, kind='mix', bg_label=-1, bg_color=(0, 0, 0)):
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()

        if kind == 'avg':
            color = image[mask].mean(axis=0)
        elif kind == 'median':
            color = np.median(image[mask], axis=0)
        elif kind == 'mix':
            std = np.std(image[mask])
            if std <= 20:
                color = image[mask].mean(axis=0)
            elif 20 < std <= 40:
                mean = image[mask].mean(axis=0)
                median = np.median(image[mask], axis=0)
                color = 0.5 * mean + 0.5 * median
            elif 40 < std:
                color = np.median(image[mask], axis=0)
        out[mask] = color
    return out


def color_ss_map(image, seg_num=200, power=1, color_space='Lab', k=10, sim_strategy='CTSF'):
    img_seg = segmentation.felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
    img_cvtcolor = label2rgb(img_seg, image, kind='mix')
    img_cvtcolor = switch_color_space(img_cvtcolor, color_space)
    S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
    S.build_regions()
    S.build_region_pairs()

    while S.num_regions() > seg_num:
        i, j = S.get_highest_similarity()
        S.merge_region(i, j)
        S.remove_similarities(i, j)
        S.calculate_similarity_for_new_region()

    image = label2rgb(S.img_seg, image, kind='mix')
    image = (image + 1) / 2
    # image = image ** power
    image = np.sign(image) * (np.abs(image))**power
    image = image / np.max(image)
    image = image * 2 - 1

    return image


# def color_ss_map(image, seg_num=200, power=1, color_space='Lab', k=10, sim_strategy='CTSF'):
#     img_seg = segmentation.felzenszwalb(image, scale=k, sigma=0.8, min_size=100)
#     img_cvtcolor = label2rgb(img_seg, image, kind='mix')
#     img_cvtcolor = switch_color_space(img_cvtcolor, color_space)
#     S = HierarchicalGrouping(img_cvtcolor, img_seg, sim_strategy)
#     S.build_regions()
#     S.build_region_pairs()
#
#     while S.num_regions() > seg_num:
#         i, j = S.get_highest_similarity()
#         S.merge_region(i, j)
#         S.remove_similarities(i, j)
#         S.calculate_similarity_for_new_region()
#
#     image = label2rgb(S.img_seg, image, kind='mix')
#     image = (image + 1) / 2
#     # image = image ** power
#     image = np.sign(image) * (np.abs(image))**power
#     image = image / np.max(image)
#     image = image * 2 - 1
#
#     return image


def selective_adacolor(batch_image, seg_num=200, power=1):
    num_job = batch_image.shape[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(color_ss_map)(np.transpose(image.detach().cpu().numpy(), (1, 2, 0)), seg_num, power) for image in batch_image)
    return torch.from_numpy(np.transpose(np.array(batch_out, dtype=np.float32), (0, 3, 1, 2))).to(device)

# def selective_adacolor(batch_image, seg_num=200, power=1):
#     batch_out = []
#     for image in batch_image:
#         # image =
#         image = color_ss_map(np.transpose(image.detach().cpu().numpy(), (1, 2, 0)), seg_num, power)
#         batch_out.append(image)
#     return torch.from_numpy(np.transpose(np.array(batch_out, dtype=np.float32), (0, 3, 1, 2))).to(device)


def simple_superpixel(batch_image, seg_num=200):
    def process_slic(image):
        seg_label = segmentation.slic(image, n_segments=seg_num, sigma=1,
                                      compactness=10, convert2lab=True, start_label=1)
        image = label2rgb(seg_label, image, kind='mix')
        # image = color.label2rgb(seg_label, image, kind='mix')
        # image = color.label2rgb(seg_label, image, kind='avg', bg_label=0)
        return image

    num_job = batch_image.shape[0]
    batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)\
                                             (np.transpose(image.detach().cpu().numpy(), (1, 2, 0))) for image in batch_image)
    # batch_out = Parallel(n_jobs=num_job)(delayed(process_slic)(image.permute(1, 2, 0)) for image in batch_image)
    return torch.from_numpy(np.transpose(np.array(batch_out, dtype=np.float32), (0, 3, 1, 2))).to(device)
    # return torch.from_numpy(np.transpose(np.array(batch_out, dtype=np.float32), (0, 3, 1, 2))).to(args.device)


def write_batch_image(image, save_dir, name, n):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fused_dir = os.path.join(save_dir, name)
    fused_image = [0] * n
    for i in range(n):
        fused_image[i] = []
        for j in range(n):
            k = i * n + j
            img = np.transpose(image[k].detach().cpu().numpy(), (1, 2, 0))
            img = (img + 1) * 127.5
            # image[k] = image[k] * 255
            fused_image[i].append(img)
        fused_image[i] = np.hstack(fused_image[i])
    fused_image = np.vstack(fused_image)
    cv2.imwrite(fused_dir, fused_image.astype(np.uint8))


if __name__ == '__main__':
    a = torch.randn((16,3,256,256))
    b = selective_adacolor(a)