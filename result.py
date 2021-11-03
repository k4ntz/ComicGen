import torch
from arguments import args, pretrain_args
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def draw_train_loss_trends(name):
    model_dir = args.save_dir + '/saved_models/'+name
    model = torch.load(model_dir, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    g_loss_totals = model['g_losses']
    d_loss_totals = model['d_losses']

    # iter_nums = np.arange(0, len(g_loss_totals)+1, 50)
    iter_nums = np.arange(0, len(g_loss_totals))

    # g_losses_ = [g_loss_totals[0]] + [g_loss_totals[i] for i in range(len(g_loss_totals) + 1) if np.mod(i, 50) == 49]
    # d_losses_ = [d_loss_totals[0]] + [d_loss_totals[i] for i in range(len(d_loss_totals) + 1) if np.mod(i, 50) == 49]

    # g_losses_ = [g_loss_totals[0]] + [g_loss_totals[i] for i in range(len(g_loss_totals) + 1)]
    # d_losses_ = [d_loss_totals[0]] + [d_loss_totals[i] for i in range(len(d_loss_totals) + 1)]

    l_g_loss_totals_ = plt.plot(iter_nums, g_loss_totals, color='deepskyblue', linestyle='-', alpha = 0.3, linewidth=1)
    l_d_loss_totals_ = plt.plot(iter_nums, d_loss_totals, color='darkorange', linestyle='-', alpha = 0.3, linewidth=1)

    # iter_nums = np.arange(0, len(g_loss_totals) + 1, 500)

    # g_losses = [g_loss_totals[0]] + [g_loss_totals[i] for i in range(len(g_loss_totals) + 1) if np.mod(i, 500) == 499]
    # d_losses = [d_loss_totals[0]] + [d_loss_totals[i] for i in range(len(d_loss_totals) + 1) if np.mod(i, 500) == 499]

    def moving_average(datas, window_size):
        i = 0
        moving_average = []
        while i < len(datas) - window_size + 1:
            this_window = datas[i:i+window_size]
            window_average = sum(this_window)/window_size
            moving_average.append(window_average)
            i += 1
        return moving_average
    g_losses = moving_average(g_loss_totals, 100)
    d_losses = moving_average(d_loss_totals, 100)
    # print(len())
    iter_nums = np.arange(0, len(g_losses))

    l_g_loss_totals = plt.plot(iter_nums, g_losses, color='deepskyblue', linestyle='-', label='Generator loss',
                               linewidth=1)
    l_d_loss_totals = plt.plot(iter_nums, d_losses, color='darkorange', linestyle='-', label='Discriminator loss',
                               linewidth=1)

    plt.xlabel('Time step')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig('train_loss_.jpg', dpi=200)
    plt.close()


def draw_pretrain_loss_trends(name):
    model_dir = pretrain_args.save_dir + '/saved_models/'+name
    model = torch.load(model_dir, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    g_losses = [model['losses'][i] for i in range(len(model['losses'])) if np.mod(i, 50) == 49]
    g_losses = [model['losses'][0]] + g_losses
    iter_nums = np.arange(0, 50001, 50)
    plt.plot(iter_nums, g_losses, color='darkorange', linestyle='-', label='recon_loss', linewidth=1)
    y_major_locator = MultipleLocator(0.05)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlabel('Time step')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(f'pretrain_g_loss.jpg', dpi=200)
    plt.show()
    plt.close()


if __name__ == "__main__":
    # draw_pretrain_loss_trends('pretrained_model_rational.pth')
    draw_train_loss_trends('rational_v10/model_rational_100000.pth')
    # print(len(range(50,99949)))