import torch
import utils
import numpy as np
import generator as G
# import discriminator as D
import discriminator_blur as D_blur
import discriminator_gray as D_gary
import loss
from surface import guided_filter
from torch import optim
from vgg import Vgg19
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
# from arguments import args
import random
from torch import nn
import argparse
from itertools import chain
from rtpt import RTPT
from pympler import muppy, summary, asizeof
import os
import ipdb


parser = argparse.ArgumentParser()
parser.add_argument("--patch_size", default=256, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--pretrain_iter", default=50000, type=int)
parser.add_argument("--train_iter", default=100000, type=int)
# parser.add_argument("--train_iter", default=40000, type=int)
parser.add_argument("--adv_train_lr", default=2e-4, type=float)
parser.add_argument("--gpu_fraction", default=0.5, type=float)
parser.add_argument("--pretrain_save_dir", default='pretrain', type=str)
parser.add_argument("--train_save_dir", default='train_cartoon', type=str)
# parser.add_argument("--use_enhance", default=True)
parser.add_argument("--use_enhance", default=False)
parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

args = parser.parse_args()

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = True


def modelsize(model, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(dataset='dataset', v=None, d=None, f=None, pretrain=True, train=True, checkpoint=False, rational=False, shared_weights=False):
    if dataset == 'dataset':
        face_photo_dir = 'dataset/photo_face'
        face_photo_list = utils.load_image_list(face_photo_dir)
        scenery_photo_dir = 'dataset/photo_scenery'
        scenery_photo_list = utils.load_image_list(scenery_photo_dir)

        face_cartoon_dir = 'dataset/cartoon_face'
        face_cartoon_list = utils.load_image_list(face_cartoon_dir)
        scenery_cartoon_dir = 'dataset/cartoon_scenery'
        scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir)
    elif dataset == 'cvpr_dataset':
        fc = 'kyoto_face'
        # fc = 'pa_face'
        sc = 'hayao'
        # sc = 'hosoda'
        # sc = 'shinkai'
        face_photo_dir = 'cvpr_dataset/photo_face'
        face_photo_list = utils.load_image_list(face_photo_dir)
        scenery_photo_dir = 'cvpr_dataset/photo_scenery'
        scenery_photo_list = utils.load_image_list(scenery_photo_dir)

        face_cartoon_dir = f'cvpr_dataset/cartoon_face/{fc}'
        face_cartoon_list = utils.load_image_list(face_cartoon_dir)
        scenery_cartoon_dir = f'cvpr_dataset/cartoon_scenery/{sc}'
        scenery_cartoon_list = utils.load_image_list(scenery_cartoon_dir)

    """
    initiate
    """

    if rational:
        # version = 'C'
        if v is None:
            version = 'A'
        else:
            version = v
        if d is None:
            degrees = (5, 4)
        else:
            degrees = d
        approx_func = "leaky_relu"

        g_net = G.Generator_Rational(shared_weights, approx_func, degrees, version, args.use_enhance).to(args.device)
        d_net_blur = D_blur.Discriminator_Rational(shared_weights, approx_func, degrees, version).to(args.device)
        d_net_gray = D_gary.Discriminator_Rational(shared_weights, approx_func, degrees, version).to(args.device)
    else:
        g_net = G.Generator(args.use_enhance).to(args.device)
        d_net_blur = D_blur.Discriminator().to(args.device)
        d_net_gray = D_gary.Discriminator().to(args.device)

    vgg = Vgg19('pretrained_models/vgg19_no_fc.npy').to(args.device)

    g_net.train()
    d_net_blur.train()
    d_net_gray.train()
    vgg.eval()

    g_optim = optim.Adam(g_net.parameters(), lr=args.adv_train_lr, betas=(0.5, 0.99))
    # d_optim = optim.Adam(list(d_net_blur.parameters()) + list(d_net_gray.parameters()), lr=args.adv_train_lr
    #                      , betas=(0.5, 0.99))
    d_optim = optim.Adam(chain(d_net_blur.parameters(), d_net_gray.parameters()), lr=args.adv_train_lr
                         , betas=(0.5, 0.99))

    # g_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=g_optim
    #                                              , milestones=[args.train_iter * 0.2, args.train_iter * 0.5]
    #                                              , gamma=0.1)
    # d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=d_optim
    #                                              , milestones=[args.train_iter * 0.2, args.train_iter * 0.5]
    #                                              , gamma=0.1)

    # if rational:
    #     pretrained_model = torch.load(f'pretrain/saved_models/pretrained_model_rational_{version}.pth', map_location=args.device)
    # else:
    #     pretrained_model = torch.load('pretrain/saved_models/pretrained_model_.pth', map_location=args.device)
    #

    l1_loss = nn.L1Loss().to(args.device)

    """
    pretrain
    """
    if rational:
        pretrain_model_dir = args.pretrain_save_dir + f'/saved_models/pretrained_model_rational_{f}_1.pth'
    else:
        pretrain_model_dir = args.pretrain_save_dir + '/saved_models/pretrained_model_v10.pth'
    if pretrain:
        # rtpt = RTPT(name_initials='SChen', experiment_name=f'Pretrain_{v}_({d[0]},{d[1]})',
        #                 max_iterations=args.pretrain_iter)

        # rtpt.start()
        g_losses = []
        for iter_num in tqdm(range(args.pretrain_iter)):
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
            recon_loss = torch.mean(F.l1_loss(photo_batch_pre, output_batch_pre))

            g_losses.append(recon_loss.detach().item())
            recon_loss.backward()
            g_optim.step()

            # rtpt.step()

            if np.mod(iter_num + 1, 50) == 0:
                print('pretrain, iter:{}, recon_loss:{}'.format(iter_num, recon_loss))
                if np.mod(iter_num + 1, 500) == 0:
                    torch.save({'epoch': iter_num,
                                'generator': g_net.state_dict(),
                                'optimizer': g_optim.state_dict(),
                                # 'loss': recon_loss,
                                'losses': g_losses
                                }, pretrain_model_dir)
                    print('pretrained model saved')

                    with torch.no_grad():
                        photo_face = utils.next_batch(face_photo_list, args.batch_size).to(args.device)
                        photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size).to(args.device)
                        result_face = g_net(photo_face)
                        result_scenery = g_net(photo_scenery)
                        utils.write_batch_image(result_face, args.pretrain_save_dir + '/images', str(iter_num) + '_face_result.jpg',
                                                4)
                        utils.write_batch_image(photo_face, args.pretrain_save_dir + '/images', str(iter_num) + '_face_photo.jpg', 4)
                        utils.write_batch_image(result_scenery, args.pretrain_save_dir + '/images',
                                                str(iter_num) + '_scenery_result.jpg', 4)
                        utils.write_batch_image(photo_scenery, args.pretrain_save_dir + '/images',
                                                str(iter_num) + '_scenery_photo.jpg', 4)

        pretrain_iter_nums = np.arange(0, len(g_losses) + 1, 50)
        pretrain_g_losses = [g_losses[0]] + [g_losses[i] for i in range(len(g_losses)) if np.mod(i, 50) == 49]
        plt.plot(pretrain_iter_nums, pretrain_g_losses, color='darkorange', linestyle='-', label='recon_loss',
                 linewidth=1)

        plt.xlabel('Time step')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        if rational:
            plt.savefig(f'pretrain_loss_rational_{f}_1.jpg', dpi=200)
        else:
            plt.savefig('pretrain_loss_10.jpg', dpi=200)
        plt.close()
    else:
        pretrained_model = torch.load(pretrain_model_dir, map_location=args.device)
        g_net.load_state_dict(pretrained_model['generator'])
        g_optim.load_state_dict(pretrained_model['optimizer'])
        print("pretrained model loaded")

        del pretrained_model

    """
    train
    """
    # start_iter = 0
    if train:
        model_dir = args.train_save_dir + '/saved_models/'

        # rtpt = RTPT(name_initials='SChen', experiment_name=f'Train_{v}_({d[0]},{d[1]})',
        #             max_iterations=args.train_iter)

        # rtpt = RTPT(name_initials='SChen', experiment_name=f'Train_v10)',
        #             max_iterations=args.train_iter)

        # rtpt.start()

        d_loss_totals = []
        g_loss_totals = []

        if checkpoint:
            model = torch.load(model_dir + f'rational_v{f}/model_rational_3500.pth', map_location=args.device)
            # model = torch.load(model_dir + 'checkpoint.pth', map_location=args.device)
            g_net.load_state_dict(model['generator'])
            d_net_blur.load_state_dict(model['discriminator_blur'])
            d_net_gray.load_state_dict(model['discriminator_gray'])
            g_optim.load_state_dict(model['g_optimizer'])
            d_optim.load_state_dict(model['d_optimizer'])
            g_loss_totals = model['g_losses']
            d_loss_totals = model['d_losses']

            start_iter = model['epoch'] + 1
            print(f'{start_iter} iterations have been trained')
            print(f'last d_loss: {d_loss_totals[-1]}')
            print(f'last g_loss: {g_loss_totals[-1]}')
            print(f'total iterations: {args.train_iter}')

        for iter_num in tqdm(range(args.train_iter)):
            """
            """
            if np.mod(iter_num, 5) == 0:
                photo_batch = utils.next_batch(face_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(face_cartoon_list, args.batch_size)
            else:
                photo_batch = utils.next_batch(scenery_photo_list, args.batch_size)
                cartoon_batch = utils.next_batch(scenery_cartoon_list, args.batch_size)

            """
            train the discriminator
            """
            d_optim.zero_grad()

            output_batch_d = g_net(photo_batch)
            output_batch_d = guided_filter(photo_batch, output_batch_d, r=1, device=args.device)

            blur_fake_d = guided_filter(output_batch_d, output_batch_d, r=5, eps=2e-1, device=args.device)
            blur_cartoon_d = guided_filter(cartoon_batch, cartoon_batch, r=5, eps=2e-1, device=args.device)

            gray_fake_d = utils.color_shift(output_batch_d)
            gray_cartoon_d = utils.color_shift(cartoon_batch)

            # loss_gray == l_texture
            real_gray_d = d_net_gray(gray_cartoon_d)
            fake_gray_d = d_net_gray(gray_fake_d)

            # loss_blur == l_surface
            real_blur_d = d_net_blur(blur_cartoon_d)
            fake_blur_d = d_net_blur(blur_fake_d)

            d_loss_gray = loss.lsgan_loss_d(real_gray_d, fake_gray_d)
            d_loss_blur = loss.lsgan_loss_d(real_blur_d, fake_blur_d)
            # d_loss_gray = loss.gan_loss_d(real_gray_d, fake_gray_d)
            # d_loss_blur = loss.gan_loss_d(real_blur_d, fake_blur_d)

            d_loss_total = d_loss_blur + d_loss_gray
            d_loss_totals.append(d_loss_total.detach().item())

            d_loss_total.backward()
            d_optim.step()
            # d_scheduler.step()

            """
            train the generator
            """
            g_optim.zero_grad()

            output_batch_g = g_net(photo_batch)
            output_batch_g = guided_filter(photo_batch, output_batch_g, r=1, device=args.device)

            if args.use_enhance:
                superpixel_batch_g = utils.selective_adacolor(output_batch_g, power=1.2)
            else:
                superpixel_batch_g = utils.simple_superpixel(output_batch_g, seg_num=200)

            vgg_photo = vgg(photo_batch)
            vgg_output = vgg(output_batch_g)
            vgg_superpixel = vgg(superpixel_batch_g)
            c, h, w = vgg_photo.shape[1:]

            # l_content
            photo_loss = l1_loss(vgg_photo, vgg_output) / (h * w * c)
            # photo_loss = torch.mean(F.l1_loss(vgg_photo, vgg_output))
            # l_structure
            superpixel_loss = l1_loss(vgg_superpixel, vgg_output) / (h * w * c)
            # superpixel_loss = torch.mean(F.l1_loss(vgg_superpixel, vgg_output))
            recon_loss = photo_loss + superpixel_loss

            # tv_loss = l_tv
            tv_loss = loss.total_variation_loss(output_batch_g)

            blur_fake_g = guided_filter(output_batch_g, output_batch_g, r=5, eps=2e-1, device=args.device)
            gray_fake_g = utils.color_shift(output_batch_g)

            fake_gray_g = d_net_gray(gray_fake_g)
            fake_blur_g = d_net_blur(blur_fake_g)

            g_loss_gray = loss.lsgan_loss_g(fake_gray_g)
            g_loss_blur = loss.lsgan_loss_g(fake_blur_g)
            # g_loss_gray = loss.gan_loss_g(fake_gray_g)
            # g_loss_blur = loss.gan_loss_g(fake_blur_g)

            # g_loss_total = 1e4 * tv_loss + 1e-1 * g_loss_blur + g_loss_gray + 2e2 * recon_loss
            g_loss_total = 1e-1 * g_loss_blur + g_loss_gray + 2e2 * recon_loss + 1e4 * tv_loss
            # g_loss_total = 1 * g_loss_blur + 10 * g_loss_gray + 2e3 * recon_loss + 1e4 * tv_loss

            g_loss_totals.append(g_loss_total.detach().item())

            # d_blur_optim.zero_grad()
            # d_gray_optim.zero_grad()
            # g_optim.zero_grad()

            g_loss_total.backward()
            g_optim.step()

            # rtpt.step()
            # g_scheduler.step()

            if np.mod(iter_num + 1, 50) == 0:
                print('Iter: {}, d_loss: {}, g_loss: {}'.format(iter_num + 1, d_loss_total, g_loss_total))
                if np.mod(iter_num + 1, 500) == 0:
                    torch.save({'epoch': iter_num,
                                'generator': g_net.state_dict(),
                                'discriminator_blur': d_net_blur.state_dict(),
                                'discriminator_gray': d_net_gray.state_dict(),
                                'g_optimizer': g_optim.state_dict(),
                                'd_optimizer': d_optim.state_dict(),
                                # 'd_blur_optimizer': d_blur_optim.state_dict(),
                                # 'd_gray_optimizer': d_gray_optim.state_dict(),
                                'g_losses': g_loss_totals,
                                'd_losses': d_loss_totals
                                }, model_dir + 'checkpoint.pth')
                    print('checkpoint saved')
                    if rational:
                        torch.save({'epoch': iter_num,
                                    'generator': g_net.state_dict(),
                                    'discriminator_blur': d_net_blur.state_dict(),
                                    'discriminator_gray': d_net_gray.state_dict(),
                                    'g_optimizer': g_optim.state_dict(),
                                    'd_optimizer': d_optim.state_dict(),
                                    # 'd_blur_optimizer': d_blur_optim.state_dict(),
                                    # 'd_gray_optimizer': d_gray_optim.state_dict(),
                                    'g_losses': g_loss_totals,
                                    'd_losses': d_loss_totals
                                    # }, model_dir + f'rational_v10/model_rational_{iter_num + 1}.pth')
                                    }, model_dir + f'rational_v{f}/model_rational_{iter_num + 1}.pth')
                        print(f'model_rational.pth saved')
                    else:
                        torch.save({'epoch': iter_num,
                                    'generator': g_net.state_dict(),
                                    'discriminator_blur': d_net_blur.state_dict(),
                                    'discriminator_gray': d_net_gray.state_dict(),
                                    'g_optimizer': g_optim.state_dict(),
                                    'd_optimizer': d_optim.state_dict(),
                                    # 'd_blur_optimizer': d_blur_optim.state_dict(),
                                    # 'd_gray_optimizer': d_gray_optim.state_dict(),
                                    'g_losses': g_loss_totals,
                                    'd_losses': d_loss_totals
                                    }, model_dir + f'model_v10/model_{iter_num + 1}.pth')
                        print(f'model.pth saved')

                    with torch.no_grad():
                        photo_face = utils.next_batch(face_photo_list, args.batch_size)
                        # cartoon_face = utils.next_batch(face_cartoon_list, args.batch_size)
                        photo_scenery = utils.next_batch(scenery_photo_list, args.batch_size)
                        # cartoon_scenery = utils.next_batch(scenery_cartoon_list, args.batch_size)
                        result_face = guided_filter(photo_face, g_net(photo_face), r=1, device=args.device)
                        result_scenery = guided_filter(photo_scenery, g_net(photo_scenery), r=1, device=args.device)
                        utils.write_batch_image(result_face, args.train_save_dir + '/images',
                                                str(iter_num) + '_face_result.jpg', 4)
                        utils.write_batch_image(photo_face, args.train_save_dir + '/images',
                                                str(iter_num) + '_face_photo.jpg', 4)
                        utils.write_batch_image(result_scenery, args.train_save_dir + '/images',
                                                str(iter_num) + '_scenery_result.jpg', 4)
                        utils.write_batch_image(photo_scenery, args.train_save_dir + '/images',
                                                str(iter_num) + '_scenery_photo.jpg',
                                                4)
                    # print('pic saved')

        train_iter_nums = np.arange(0, len(g_loss_totals) + 1, 50)
        train_g_losses = [g_loss_totals[0]] + [g_loss_totals[i] for i in range(len(g_loss_totals)) if
                                               np.mod(i, 50) == 49]
        train_d_losses = [d_loss_totals[0]] + [d_loss_totals[i] for i in range(len(d_loss_totals)) if
                                               np.mod(i, 50) == 49]
        plt.plot(train_iter_nums, train_g_losses, color='deepskyblue', linestyle='-', label='g_loss_total', linewidth=1)
        plt.plot(train_iter_nums, train_d_losses, color='darkorange', linestyle='-', label='d_loss_total', linewidth=1)

        plt.xlabel('Time step')
        plt.ylabel('Loss')
        plt.legend(loc='best')
        if rational:
            plt.savefig(f'train_loss_rational_v{f}.jpg', dpi=200)
        else:
            plt.savefig('train_loss_v10.jpg', dpi=200)
        plt.close()


if __name__ == '__main__':
    # args = arg_parser()
    # setup_seed(42)
    # for v, f in (('A', 13), ('B', 14), ('C', 15), ('D', 16)):
        # main("dataset", pretrain=False, checkpoint=False, rational=True, norm=False)
        # main("dataset", v=v, f=f, pretrain=False, checkpoint=False, rational=True, norm=False)
    # for v, f in (('A', 13), ('B', 14), ('C', 15), ('D', 16)):
    #     main("dataset", v=v, f=f, pretrain=True, train=False, checkpoint=False, rational=True, norm=False)
    # for v, f in (('A', 13), ('B', 14), ('C', 15), ('D', 16)):
    #     main("dataset", v=v, f=f, pretrain=False, train=True, checkpoint=False, rational=True, norm=False)

    # for v, d, f in (('A', (7, 6), 29),
    #                 ('C', (3, 2), 23), ('C', (7, 6), 31), ('D', (3, 2), 24), ('D', (7, 6), 32)):
        # main("dataset", v=v, d=d, f=f, pretrain=True, train=True, checkpoint=False, rational=True, norm=False)
    main("dataset", pretrain=True, train=True, checkpoint=False, rational=True, shared_weights=False)
    # main("dataset", v='B', d=(7, 6), f=30, pretrain=False, train=True, checkpoint=False, rational=True, norm=False)

    # for v, d, f in (('A', (3, 2), 17), ('A', (7, 6), 25),
    #                 ('C', (3, 2), 19), ('C', (7, 6), 27), ('D', (3, 2), 20), ('D', (7, 6), 28)):
    #     main("dataset", v=v, d=d, f=f, pretrain=True, train=True, checkpoint=False, rational=True, norm=False)
    # for v, d, f in (('B', (3, 2), 18), ('B', (7, 6), 26)):
    #     main("dataset", v=v, d=d, f=f, pretrain=True, train=True, checkpoint=False, rational=True, norm=False)

