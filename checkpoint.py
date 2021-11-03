import torch
from arguments import args, pretrain_args


def check(pretrain=False, rational=False):
    if pretrain:
        if rational:
            pretrained_model_dir = pretrain_args.save_dir + '/saved_models/pretrained_model_rational.pth'

            pretrained_model = torch.load(pretrained_model_dir, map_location=torch.device('cpu'))

            pretrained_g_losses = pretrained_model['losses']
            pretrained_start_iter = pretrained_model['epoch'] + 1

            print(f'pretrained_g_losses accounts: {len(pretrained_g_losses)}')
            print(f"pretrained_start_iterations: {pretrained_start_iter}")
        else:
            pretrained_model_dir = pretrain_args.save_dir + '/saved_models/pretrained_model.pth'

            pretrained_model = torch.load(pretrained_model_dir, map_location=torch.device('cpu'))

            pretrained_g_losses = pretrained_model['losses']
            pretrained_start_iter = pretrained_model['epoch'] + 1

            print(f'pretrained_g_losses accounts: {len(pretrained_g_losses)}')
            print(f"pretrained_start_iterations: {pretrained_start_iter}")
    else:
        if rational:
            model_dir = args.save_dir + '/saved_models/model_rational.pth'

            model = torch.load(model_dir, map_location=torch.device('cpu'))

            g_loss_totals = model['g_losses']
            d_loss_totals = model['d_losses']
            start_iter = model['epoch'] + 1

            print(f'g_losses accounts: {len(g_loss_totals)}')
            print(f'd_losses accounts: {len(d_loss_totals)}')
            print(f"start_iterations: {start_iter}")
        else:
            model_dir = args.save_dir + '/saved_models/model.pth'

            model = torch.load(model_dir, map_location=torch.device('cpu'))

            g_loss_totals = model['g_losses']
            d_loss_totals = model['d_losses']
            start_iter = model['epoch'] + 1

            print(f'g_losses accounts: {len(g_loss_totals)}')
            print(f'd_losses accounts: {len(d_loss_totals)}')
            print(f"start_iterations: {start_iter}")


if __name__ == '__main__':
    # check(pretrain=False, rational=False)
    model_dir = args.save_dir + '/saved_models/rational_v30/model_rational_25000.pth'
    model = torch.load(model_dir, map_location=torch.device('cpu'))
    generator = model['generator']
    for k,v in generator.items():
        if 'ator' in k:
            print(k, v)
            print(" ")
        # print(k, v.shape)




