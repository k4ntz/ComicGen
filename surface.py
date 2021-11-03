import numpy as np
import torch
import matplotlib.pyplot as plt
# from arguments import args
import torch.nn.functional as F

# H = 128
# W = 128
# C = 3


def load_image(path):
    return plt.imread(path)


def display_image(img):
    """ Show an image with matplotlib:

    Args:
        Image as numpy array (H,W,3)
    """

    #
    # You code here
    #
    # display a show
    plt.imshow(img)
    # display a figure
    plt.show()
    # close a figure window.
    plt.close()


# def diff_x(input, r):
#     assert input.dim() == 4
#
#     left   = input[:, :,         r:2 * r + 1]
#     middle = input[:, :, 2 * r + 1:         ] - input[:, :,           :-2 * r - 1]
#     right  = input[:, :,        -1:         ] - input[:, :, -2 * r - 1:    -r - 1]
#
#     output = torch.cat([left, middle, right], dim=2)
#
#     return output
#
#
# def diff_y(input, r):
#     assert input.dim() == 4
#
#     left   = input[:, :, :,         r:2 * r + 1]
#     middle = input[:, :, :, 2 * r + 1:         ] - input[:, :, :,           :-2 * r - 1]
#     right  = input[:, :, :,        -1:         ] - input[:, :, :, -2 * r - 1:    -r - 1]
#
#     output = torch.cat([left, middle, right], dim=3)
#
#     return output


# def box_filter(x, r):
#     return diff_y(diff_x(x.cumsum(dim=2), r).cumsum(dim=3), r)

def box_filter(x, r, device):
    k_size = int(2*r+1)
    ch = x.shape[1]
    weight = 1/(k_size**2)
    box_kernel = weight*torch.ones((ch, 1, k_size, k_size), dtype=torch.float32).to(device)
    # box_kernel = weight * np.ones((k_size, k_size, ch, 1))
    # box_kernel = box_kernel.double()
    output = F.conv2d(x, box_kernel, padding=r, groups=ch)
    # print(box_kernel)
    return output


def guided_filter(x, y, r, device, eps=1e-2):
    x_N, x_C, x_H, x_W = x.shape
    # y_N, y_C, y_H, y_W = y.shape
    N = box_filter(torch.ones(1, 1, x_H, x_W, dtype=torch.float32).to(device), r, device)
    # N = box_filter(Variable(x.data.new().resize_((1,1,x_H,x_W)).fill_(1.0)),r)
    # print(N)

    mean_x = box_filter(x, r, device) / N
    mean_y = box_filter(y, r, device) / N
    cov_xy = box_filter(x * y, r, device) / N - mean_x * mean_y
    var_x = box_filter(x * x, r, device) / N - mean_x * mean_x

    A = cov_xy / (var_x + eps)
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r, device) / N
    mean_b = box_filter(b, r, device) / N

    return mean_A * x + mean_b


if __name__ == '__main__':
    x = torch.from_numpy(np.random.random((1,3,7,7))).float()
    y = torch.from_numpy(np.random.random((1,3,7,7))).float()
    print(x.dtype)
    print(y.dtype)
    m = guided_filter(x,y,1,'cpu')
    print(m.shape)
