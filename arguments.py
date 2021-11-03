import argparse
import torch


def train_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--total_iter", default=100000, type=int)
    parser.add_argument("--adv_train_lr", default=2e-4, type=float)
    parser.add_argument("--gpu_fraction", default=0.5, type=float)
    parser.add_argument("--save_dir", default='train_cartoon', type=str)
    parser.add_argument("--use_enhance", default=True)
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()
    return args


def pretrain_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch_size", default=256, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--total_iter", default=50000, type=int)
    parser.add_argument("--adv_train_lr", default=2e-4, type=float)
    parser.add_argument("--gpu_fraction", default=0.5, type=float)
    parser.add_argument("--save_dir", default='pretrain', type=str)
    parser.add_argument("--device", default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    args = parser.parse_args()
    return args


args = train_arg_parser()
pretrain_args = pretrain_arg_parser()
