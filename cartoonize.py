import generator as G
import os
import cv2
import torch
from tqdm import tqdm
import numpy as np
from arguments import args
from surface import guided_filter
import torch.nn.functional as F
from torch import nn
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"


def resize_crop(image):
    h, w, c = np.shape(image)
    if min(h, w) > 720:
        if h > w:
            h, w = int(720 * h / w), 720
        else:
            h, w = 720, int(720 * w / h)
    image = cv2.resize(image, (w, h),
                       interpolation=cv2.INTER_AREA)
    h, w = (h // 8) * 8, (w // 8) * 8
    image = image[:h, :w, :]
    return image


def transformer(img):
    max_ = torch.max(img)
    min_ = torch.min(img)
    img = (img - min_) / (max_ - min_)
    return img * 2 - 1


def single_cartoonize(load_folder, target_folder, save_folder, model_name, shared=False, rational=False, v=None, d=None):
    # version = 6.2
    # model_path = args.save_dir + f'/saved_models/model_v{version}.pth'
    # name = 'model_rational_10000.pth'
    print(model_name)
    model_path = f'models/' + model_name

    with torch.no_grad():
        if rational:
            approx_func = 'leaky_relu'
            degrees = d
            version = v
            generator = G.Generator_Rational(shared, approx_func, degrees, version, args.use_enhance).to(device)
        else:
            generator = G.Generator(args.use_enhance).to(device)
        # generator = Generator().to(args.device)
        generator.eval()
        model = torch.load(model_path, map_location=device)
        generator.load_state_dict(model['generator'])

        # transformer = transforms.Compose([
        #     # transforms.Resize((256, 256)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        # ])

        # print(generator.state_dict()['conv_32_3_7_1.bias'])
        name_list = [name for name in os.listdir(load_folder)]

        for name in tqdm(name_list):
            try:
                load_path = os.path.join(load_folder, name)
                save_path = os.path.join(save_folder, name)
                image = cv2.imread(load_path)
                image = resize_crop(image)

                # target_image = cv2.imread(os.path.join(target_folder, name))
                # target_image = resize_crop(target_image)

                batch_image = image.astype(np.float32) / 127.5 - 1
                batch_image = np.transpose(batch_image, (2, 0, 1))
                batch_image = torch.from_numpy(np.expand_dims(batch_image, axis=0)).to(device)
                output = generator(batch_image)
                output = guided_filter(batch_image, output, r=1, eps=5e-3, device=device)
                # output = torch.clamp(output, -1, 1)
                output = output.permute(0, 2, 3, 1)
                output = np.squeeze(output.cpu().numpy())
                output = (output+1)*127.5
                output = np.clip(output, 0, 255).astype(np.uint8)
                # img = np.concatenate((image, target_image, output), axis=1)
                # cv2.imwrite(save_path, img)
                cv2.imwrite(save_path, output)
            except:
                print('cartoonize {} failed'.format(os.path.join(load_folder, name)))


if __name__ == '__main__':
    load_folder = 'images'
    compared_folder = 'target_images'
    save_folder = 'cartoonized_images'
    # load_folder = 'images/AIML_Lab'
    # save_folder = 'cartoonized_images/AIML_Lab'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    single_cartoonize(load_folder, compared_folder, save_folder, 'model_rats.pth',True,True,'B',(7,6))
    # folder_pairs = [('images', 'target_images', 'cartoonized_images'),('images/AIML_Lab', 'target_images/AIML_Lab', 'cartoonized_images/AIML_Lab')]

    # single_cartoonize(load_folder, save_folder, 'checkpoint.pth', rational=True)
    # for l, t, s in folder_pairs:
    #     if not os.path.exists(s):
    #         os.mkdir(s)
    #     single_cartoonize(l, t, s, 'rational_v22/model_rational_40000.pth', rational=True, v='B', d=(3,2))
        # single_cartoonize(l, t, s, 'model_v4.pth', rational=False, version='A')
    # series_cartoonize(load_folder, save_folder, rational=False)
    # continue_cartoonize(load_folder, save_folder, rational=False)
