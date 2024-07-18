import argparse
from math import pi
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_source_dir', default=r'D:\.../')
    parser.add_argument('--train_target_dir', default=r'D:\.../')

    parser.add_argument('--log_dir', default='')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_epochs', default=200, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--resume', default=False)
    parser.add_argument('--model_chk_iter', default=1, type=int)
    parser.add_argument('--lpips', default='vgg')
    parser.add_argument('--load_mode', default='train_main')

    # network type
    parser.add_argument("--norm_use", default=True, type=bool)
    parser.add_argument("--lrelu_use", default=True, type=bool)
    parser.add_argument("--lrelu_slope", default=0.1, type=float)
    parser.add_argument("--batch_mode", default='B', type=str)
    parser.add_argument("--zero_padding", default=True, type=bool)
    parser.add_argument('--initial_channel', default=8, type=int)
    parser.add_argument('--output_channel', default=1, type=int)
    # hyper-parameter
    parser.add_argument("--lr_disc", default=1e-4, type=float)
    parser.add_argument("--lr_gen", default=1e-4, type=float)
    parser.add_argument("--lr_decay_epoch", default=2, type=int)
    parser.add_argument("--lr_decay_rate", default=0.8, type=float)
    parser.add_argument("--distance_regularizer", default=100, type=float)
    parser.add_argument("--penalty_regularizer", default=20, type=int)
    parser.add_argument("--cycle_regularizer", default=100, type=int)
    parser.add_argument("--gan_regularizer", default=1, type=int)
    parser.add_argument("--diffraction_regularizer", default=100, type=float)
    parser.add_argument("--ssim_regularizer", default=20, type=float)
    parser.add_argument("--lpips_G_regularizer", default=100, type=float)
    parser.add_argument("--wiener_gamma", default=1e5, type=float)

    parser.add_argument('--retrain', default=False)
    parser.add_argument('--checkpoint', default=r'E:\.../')
    parser.add_argument('--shuffle', default=False)


    return parser.parse_args()
average_pool = nn.AvgPool2d(2, stride=2)


