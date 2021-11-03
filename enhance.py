import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
import cv2
import os


def image_enhance(image):
    return 0


if __name__ == '__main__':
    folder = 'v5_100000'
    load_folder = f'cartoonized_images/{folder}'
    save_folder = 'enhanced_images'

    filename = 'hessenSchaftWissen2019.jpg'

    load_path = os.path.join(load_folder, filename)
    save_path = os.path.join(save_folder, filename)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    image = cv2.imread(load_folder)

    enhanced = image_enhance(image)


