import torch
import utils
import cv2
import numpy as np
from surface import guided_filter
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "representations"

images_list = list()
for name in os.listdir(path):
    images_list.append(name)
# images_list = utils.load_image_list(path)
for name in images_list:
    image = cv2.imread(os.path.join(path, name))
    batch_image = image.astype(np.float32) / 127.5 - 1
    batch_image = np.transpose(batch_image, (2, 0, 1))
    batch_image = torch.from_numpy(np.expand_dims(batch_image, axis=0)).to(device)

    """
    surface representations
    # """
    surface_representations = guided_filter(batch_image, batch_image, r=5, eps=2e-1, device=device)
    surface_representations = surface_representations.permute(0, 2, 3, 1)
    surface_representations = np.squeeze(surface_representations.cpu().numpy())
    surface_representations = (surface_representations + 1) * 127.5
    surface_representations = np.clip(surface_representations, 0, 255).astype(np.uint8)
    # cv2.imshow("a", surface_representations)
    # output = np.concatenate((image, surface_representations), axis=1)
    cv2.imwrite(os.path.join(path, "surface_"+name), surface_representations)

    """
    texture representations
    """
    texture_representations = utils.color_shift(batch_image)
    texture_representations = texture_representations.permute(0, 2, 3, 1)
    texture_representations = np.squeeze(texture_representations.cpu().numpy(), axis=0)
    texture_representations = (texture_representations + 1) * 127.5
    texture_representations = np.clip(texture_representations, 0, 255).astype(np.uint8)
    # output = np.concatenate((image, texture_representations), axis=1)
    cv2.imwrite(os.path.join(path, "texture_" + name), texture_representations)

    "structure representations"
    structure_representations = utils.selective_adacolor(batch_image, power=1.2)
    # structure_representations = utils.simple_superpixel(batch_image, seg_num=200)
    structure_representations = structure_representations.permute(0, 2, 3, 1)
    structure_representations = np.squeeze(structure_representations.cpu().numpy(), axis=0)
    structure_representations = (structure_representations + 1) * 127.5
    structure_representations = np.clip(structure_representations, 0, 255).astype(np.uint8)
    # output = np.concatenate((image, texture_representations), axis=1)
    cv2.imwrite(os.path.join(path, "structure_" + name), structure_representations)

    simple_structure_representations = utils.simple_superpixel(batch_image, seg_num=200)
    simple_structure_representations = simple_structure_representations.permute(0, 2, 3, 1)
    simple_structure_representations = np.squeeze(simple_structure_representations.cpu().numpy(), axis=0)
    simple_structure_representations = (simple_structure_representations + 1) * 127.5
    simple_structure_representations = np.clip(simple_structure_representations, 0, 255).astype(np.uint8)
    #
    output = np.concatenate((image,structure_representations, simple_structure_representations), axis=1)
    cv2.imwrite(os.path.join(path, "simple_structure_" + name), simple_structure_representations)
    cv2.imwrite(os.path.join(path, "concate_" + name), output)



