import torch
import utils
import numpy as np
import generator as G
import torch.nn.functional as F
from tqdm import tqdm
# from arguments import pretrain_args as args
import random
import argparse
from torch import nn
from vgg import Vgg19
import matplotlib.pyplot as plt
from rtpt import RTPT


parser = argparse.ArgumentParser()
parser.add_argument("--patch_size", default=256, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--pretrain_iter", default=50000, type=int)
parser.add_argument("--adv_train_lr", default=2e-4, type=float)
parser.add_argument("--gpu_fraction", default=0.5, type=float)
parser.add_argument("--save_dir", default='pretrain', type=str)
parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

args = parser.parse_args()

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def pretrain(approx_func, degrees, version, checkpoint=False, rational=False):
    face_photo_dir = 'dataset/photo_face'
    face_photo_list = utils.load_image_list(face_photo_dir)
    scenery_photo_dir = 'dataset/photo_scenery'
    scenery_photo_list = utils.load_image_list(scenery_photo_dir)

    rtpt = RTPT(name_initials='SChen', experiment_name='ComicGen',
                max_iterations=args.pretrain_iter)
    rtpt.start()

    if rational:
        model_dir = args.save_dir + f'/saved_models/pretrained_model_rational_{version}_0.pth'
        g_net = G.Generator_Rational(approx_func=approx_func, degrees=degrees, version=version).to(args.device)

    else:
        model_dir = args.save_dir+'/saved_models/pretrained_model_.pth'
        g_net = G.Generator().to(args.device)

    g_optim = torch.optim.Adam(g_net.parameters(), lr=args.adv_train_lr, betas=(0.5, 0.99))
    g_net.train()

    # vgg_model = Vgg19('pretrained_models/vgg19_no_fc.npy').to(args.device)
    # vgg_model.eval()

    g_losses = []
    start_iter = 0

    l1_loss = nn.L1Loss().to(args.device)

    if checkpoint:
        model = torch.load(model_dir)
        g_net.load_state_dict(model['generator'])
        g_optim.load_state_dict(model['optimizer'])
        start_iter = model['epoch'] + 1
        g_losses = model['losses']

    for iter_num in tqdm(range(start_iter, args.pretrain_iter)):
        if np.mod(iter_num, 5) == 0:
            photo_batch_pre = utils.next_batch(face_photo_list, args.batch_size)
        else:
            photo_batch_pre = utils.next_batch(scenery_photo_list, args.batch_size)

        g_optim.zero_grad()

        # photo_feature_pre = vgg(photo_batch_pre)
        # output_batch_pre = g_net(photo_batch_pre)
        # g_feature_pre = vgg(output_batch_pre)
        #
        # recon_loss = 10 * l1_loss(g_feature_pre, photo_feature_pre.detach())

        output_batch_pre = g_net(photo_batch_pre)
        recon_loss = torch.mean(l1_loss(photo_batch_pre, output_batch_pre))

        g_losses.append(recon_loss.detach().item())
        recon_loss.backward()
        g_optim.step()

        if np.mod(iter_num+1, 50) == 0:
            print('pretrain, iter:{}, recon_loss:{}'.format(iter_num, recon_loss))
            if np.mod(iter_num+1, 500) == 0:
                torch.save({'epoch': iter_num,
                            'generator': g_net.state_dict(),
                            'optimizer': g_optim.state_dict(),
                            # 'loss': recon_loss,
                            'losses': g_losses
                            }, model_dir)
                print('pretrained model saved')
        rtpt.step()

    pretrain_iter_nums = np.arange(0, len(g_losses)+1, 50)
    pretrain_g_losses = [g_losses[0]] + [g_losses[i] for i in range(len(g_losses)) if np.mod(i, 50) == 49]
    plt.plot(pretrain_iter_nums, pretrain_g_losses, color='darkorange', linestyle='-', label='recon_loss', linewidth=1)

    plt.xlabel('Time step')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    if rational:
        plt.savefig(f'pretrain_loss_rational_{version}.jpg', dpi=200)
    else:
        plt.savefig('pretrain_loss.jpg', dpi=200)
    plt.close()


if __name__ == "__main__":
    approx_func = "leaky_relu"
    degrees = (5, 4)
    # version = "A"
    for version in ['A', 'B', 'C', 'D']:
        pretrain(approx_func, degrees, version, checkpoint=False, rational=True)


