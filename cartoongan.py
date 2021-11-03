import generator as G
import discriminator_blur as D_blur
import discriminator_gray as D_gary
from torch import optim
from vgg import Vgg19
import torch
from surface import guided_filter
import loss
import utils
import torch.nn.functional as F
from arguments import args


class CartoonGAN():
    def __init__(self):
        self.g_net = G.Generator().to(args.device)
        self.d_net_blur = D_blur.Discriminator().to(args.device)
        self.d_net_gray = D_gary.Discriminator().to(args.device)
        self.g_optim = optim.Adam(self.g_net.parameters(), lr=args.adv_train_lr, betas=(0.5, 0.99))
        self.d_blur_optim = optim.Adam(self.d_net_blur.parameters(), lr=args.adv_train_lr, betas=(0.5, 0.99))
        self.d_gray_optim = optim.Adam(self.d_net_gray.parameters(), lr=args.adv_train_lr, betas=(0.5, 0.99))
        self.vgg_model = Vgg19('saved_models/vgg19_no_fc.npy')
        # self.vgg_model = self.load_vgg('saved_models/vgg19_no_fc.npy')
        self.load_pretrained_model('pretrain/saved_models/pretrained_model.pth')

    def load_pretrained_model(self, path):
        pretrained_model = torch.load(path)
        self.g_net.load_state_dict(pretrained_model['generator'])
        self.g_optim.load_state_dict(pretrained_model['optimizer'])
        print('pre-trained model loaded')

    # def load_vgg(self, vgg19_npy_path):
    #     def load_model(path):
    #         np_vgg = np.load(path, encoding='latin1', allow_pickle=True).item()
    #         data_dict = {}
    #         for k, [w, b] in np_vgg.items():
    #             if k in ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
    #                      'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4']:
    #                 data_dict[k + '.weight'] = torch.as_tensor(np.transpose(w, (3, 2, 0, 1)))
    #                 data_dict[k + '.bias'] = torch.as_tensor(b)
    #         return data_dict
    #     model = Vgg19()
    #     model.load_state_dict(load_model(vgg19_npy_path))
    #     return model.to(args.device)

    def generate_images(self, inputs):
        outputs = self.g_net(inputs)
        outputs = guided_filter(inputs, outputs, r=1)
        return outputs

    def train_step_generator(self, tv_loss, loss_blur, loss_gray, input_batch, output_batch):
        if args.use_enhance:
            superpixel_batch = utils.selective_adacolor(output_batch, power=1.0)
        else:
            superpixel_batch = utils.simple_superpixel(output_batch, seg_num=200)

        # vgg_photo = self.vgg_model(input_batch)
        # vgg_output = self.vgg_model(output_batch)
        # vgg_superpixel = self.vgg_model(superpixel_batch)
        vgg_photo = self.vgg_model.build_conv4_4(input_batch)
        vgg_output = self.vgg_model.build_conv4_4(output_batch)
        vgg_superpixel = self.vgg_model.build_conv4_4(superpixel_batch)
        c, h, w = vgg_photo.shape[1:]

        photo_loss = torch.mean(F.l1_loss(vgg_photo, vgg_output)) / (h * w * c)
        superpixel_loss = torch.mean(F.l1_loss(vgg_superpixel, vgg_output)) / (h * w * c)
        recon_loss = photo_loss + superpixel_loss
        g_loss_total = 1e4 * tv_loss + 1e-1 * loss_blur + loss_gray + 2e2 * recon_loss
        # self.g_optim.zero_grad()
        # g_loss_total.backward()
        # self.g_optim.step()
        return g_loss_total

    def train_step_discriminator(self, blur_fake, blur_cartoon, gray_fake, gray_cartoon):
        # loss_gray == l_texture
        d_loss_gray, g_loss_gray = loss.lsgan_loss(self.d_net_gray, gray_cartoon, gray_fake)
        # loss_blur == l_surface
        d_loss_blur, g_loss_blur = loss.lsgan_loss(self.d_net_blur, blur_cartoon, blur_fake)

        d_loss_total = d_loss_blur + d_loss_gray
        # self.d_blur_optim.zero_grad()
        # self.d_gray_optim.zero_grad()
        # d_loss_total.backward(retain_graph=True)
        # self.d_blur_optim.step()
        # self.d_gray_optim.step()
        return d_loss_total, g_loss_blur, g_loss_gray

