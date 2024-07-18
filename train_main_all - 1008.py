##
## The Field_G, Discriminator, Distance_Generator and functions were adapted from https://github.com/csleemooo/Deep_learning_based_on_parameterized_physical_forward_model_for_adaptive_holographic_imaging
## MDGAN was inspired by their works and used for passive lensless imaging system.
## Readers can adopt such architecture to address problems in their research domains, where the generator and discriminator can take any form. Here, we utilized existing networks from the website above.

from dataload import *
from netmodel import *
import torch
import torch.nn.parallel
import torch.nn as nn
from Initialization_experiment import parse_args
from itertools import chain
import skimage.io
from torch.autograd import Variable
import numpy as np
from Inverse_operator import Distance_Generator, Field_Generator, Discriminator, WienerBlock, Imaging_fft, normalization
from functions.SSIM import SSIM
from functions.gradient_penalty import calc_gradient_penalty
from torchvision.models import vgg16
import lpips
import os
import matplotlib.pyplot as plt
# from pytorch_msssim import ssim
import itertools
from psf_generator import *
def make_path(path):
    import os
    if not os.path.isdir(path):
        os.mkdir(path)
args = parse_args()
dtype = torch.cuda.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
save_path = 'E:/NC/nc'
make_path(os.path.join(save_path, 'save'))

# load data
data = get_dataloaders(args)
# load model
diffraction_G = Field_Generator(args).to(device=device)
Field_G = Field_Generator(args).to(device=device)
distance_G = Distance_Generator(args).to(device=device)
Field_D = Discriminator(args).to(device=device)
diffraction_D = Discriminator(args).to(device=device)

wiener = WienerBlock(args).to(device=device)
average_pool = nn.AvgPool2d(2, stride=2)

distance_G = nn.DataParallel(distance_G)
Field_G = nn.DataParallel(Field_G)
diffraction_G = nn.DataParallel(diffraction_G)
Field_D = nn.DataParallel(Field_D)
diffraction_D = nn.DataParallel(diffraction_D)
# optimizer
op_G = torch.optim.Adam(itertools.chain(Field_G.parameters(), diffraction_G.parameters(), distance_G.parameters()), lr=args.lr_gen, betas=(0.5, 0.999))
op_D = torch.optim.Adam(itertools.chain(Field_D.parameters(), diffraction_D.parameters()), lr=args.lr_disc, betas=(0.5, 0.999))
# op_D = torch.optim.Adam(Field_D.parameters(), lr=args.lr_disc, betas=(0.5, 0.999))
# scheduler
lr_scheduler_G = torch.optim.lr_scheduler.StepLR(op_G, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)
lr_scheduler_D = torch.optim.lr_scheduler.StepLR(op_D, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)
# loss
criterion_cycle = nn.L1Loss()
criterion_MSE = nn.MSELoss()
criterion_wgan = torch.mean
criterion_ssim = SSIM()
loss_fn_lpips = lpips.LPIPS(net=args.lpips).to(device)
start_epoch = 0

if args.retrain:
    if args.checkpoint:
        checkpoint = os.path.join(args.checkpoint)
        ckpt = torch.load(checkpoint + '/iterations_200/model.pth')  # 这里需要实时更改从哪里读入
        distance_G.load_state_dict(ckpt['distance_G_state_dict'])
        diffraction_G.load_state_dict(ckpt['diffraction_G_state_dict'])
        Field_G.load_state_dict(ckpt['Field_G_state_dict'])
        Field_D.load_state_dict(ckpt['Field_D_state_dict'])
        diffraction_D.load_state_dict(ckpt['diffraction_D_state_dict'])
        start_epoch = ckpt['iteration']

        print('Loaded checkpoint from:' + checkpoint + '/iterations_200/model.pth')
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
# run training epochs
print('=> starting training')
# train_main
for epoch in range(start_epoch, args.num_epochs):
    # print('epoch: {}, lr: {}'.format(epoch, optimizer.param_groups[0]['lr']))
    diffraction_G.train()
    Field_G.train()
    distance_G.train()
    Field_D.train()
    diffraction_D.train()
    # diffraction_D.train()
    for i, batch in enumerate(data):
        # 第i个batch
        diffraction, real_amplitude, real_distance = batch
        diffraction = normalization(diffraction).to(device)
        real_amplitude = normalization(average_pool(real_amplitude)).to(device)
        real_distance = ((real_distance - 100) / 550).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
        # rand_distance = 25*(torch.rand(size=(args.batch_size, 1, 1, 1)).to(device=device).float()-0.5)
        # top cycle
        fake_distance = distance_G(diffraction)
        psf_guess = generate_psf(fake_distance)
        diffraction_prepro = wiener(diffraction, psf_guess)
        fake_amplitude = Field_G(diffraction_prepro)[:, :, 384:384 + 256, 384:384 + 256]
        consistency_diffraction = diffraction_G(Imaging_fft(fake_amplitude, psf_guess))

        # bottom cycle
        psf_bottom = generate_psf(real_distance)
        # psf_bottom =psf_guess
        fake_diffraction = diffraction_G(Imaging_fft(real_amplitude, psf_bottom))
        consistency_distance = distance_G(fake_diffraction)
        psf_consistency = generate_psf(consistency_distance)
        fake_diffraction_prepro = wiener(fake_diffraction, psf_consistency)
        consistency_amplitude = Field_G(fake_diffraction_prepro)[:, :, 384:384 + 256, 384:384 + 256]

        if args.gan_regularizer:
            op_D.zero_grad()
            # Field_D_loss
            fake_amplitude_D = Field_D(fake_amplitude.detach())
            real_amplitude_D = Field_D(real_amplitude)
            amplitude_D_penalty_loss = calc_gradient_penalty(Field_D, real_amplitude, fake_amplitude, real_amplitude.shape[0])
            amplitude_D_adversarial_loss = criterion_wgan(fake_amplitude_D.mean(dim=(-2, -1))) - criterion_wgan(
                real_amplitude_D.mean(dim=(-2, -1)))
            loss_field_D = amplitude_D_penalty_loss + amplitude_D_adversarial_loss
            # diffraction_D_loss
            fake_D = diffraction_D(fake_diffraction.detach())
            real_D = diffraction_D(diffraction)
            diffraction_D_penalty_loss = calc_gradient_penalty(diffraction_D, diffraction, fake_diffraction, diffraction.shape[0])
            diffraction_D_adversarial_loss = criterion_wgan(fake_D.mean(dim=(-2, -1))) - criterion_wgan(
                real_D.mean(dim=(-2, -1)))
            loss_diffraction_D = diffraction_D_penalty_loss + diffraction_D_adversarial_loss
            # backward netD
            loss_D = args.penalty_regularizer * (loss_field_D + loss_diffraction_D)
            loss_D.backward()
            # loss_field_D.backward()
            op_D.step()


        ## train  generator
        op_G.zero_grad()
        # D_loss_amplitude = -criterion_wgan(Field_D(fake_amplitude).mean(dim=(-2, -1))) - args.penalty_regularizer * calc_gradient_penalty(Field_D, real_amplitude, fake_amplitude, real_amplitude.shape[0])
        # D_loss_diffraction = -criterion_wgan(diffraction_D(fake_diffraction).mean(dim=(-2, -1))) - args.penalty_regularizer * calc_gradient_penalty(diffraction_D, diffraction, fake_diffraction, diffraction.shape[0])
        D_loss_amplitude = criterion_wgan(Field_D(real_amplitude).mean(dim=(-2, -1)))-criterion_wgan(Field_D(fake_amplitude).mean(dim=(-2, -1))) - calc_gradient_penalty(Field_D, real_amplitude, fake_amplitude, real_amplitude.shape[0])
        D_loss_diffraction = criterion_wgan(diffraction_D(diffraction).mean(dim=(-2, -1)))-criterion_wgan(diffraction_D(fake_diffraction).mean(dim=(-2, -1))) - calc_gradient_penalty(diffraction_D, diffraction, fake_diffraction, diffraction.shape[0])
        cycle_loss_diffraction = args.diffraction_regularizer * criterion_cycle(consistency_diffraction, diffraction) + args.diffraction_regularizer * (1-criterion_ssim(consistency_diffraction, diffraction))
        cycle_loss_amplitude = args.ssim_regularizer * criterion_cycle(consistency_amplitude, real_amplitude) + args.ssim_regularizer *(1-criterion_ssim(consistency_amplitude, real_amplitude))
        cycle_loss_distance = args.distance_regularizer * (criterion_cycle(consistency_distance, real_distance) + criterion_cycle(fake_distance, real_distance))
        lpisp_G = 100 * torch.mean(loss_fn_lpips(fake_amplitude, real_amplitude))
        # loss_fake = 100 * criterion_cycle(fake_amplitude, real_amplitude) + args.ssim_regularizer *(1-criterion_ssim(fake_amplitude, real_amplitude)) + 100 * torch.mean(loss_fn_lpips(fake_amplitude, real_amplitude))
        loss_fake_diffraction = args.diffraction_regularizer * criterion_cycle(fake_diffraction, diffraction)
        # loss_G = cycle_loss_amplitude + cycle_loss_diffraction + cycle_loss_distance + cycle_loss_amplitude_ssim + cycle_loss_diffraction_ssim + D_loss_amplitude + D_loss_diffraction
        loss_G = args.penalty_regularizer * (D_loss_amplitude+D_loss_diffraction) + cycle_loss_diffraction \
                 + cycle_loss_amplitude + cycle_loss_distance + lpisp_G + loss_fake_diffraction
        # loss_top.backward(retain_graph=True)
        # loss_bottom.backward(retain_graph=True)
        loss_G.backward()
        op_G.step()

        print(
            f"epoch{epoch},{i}, "
            f"rand_distance_l1: {cycle_loss_distance}, lpisp_G: {lpisp_G}, "
            f"D_loss_diffraction: {args.penalty_regularizer * D_loss_diffraction}, "
            f"D_loss_amplitude: {args.penalty_regularizer * D_loss_amplitude} "
        )

        if i == 0:
            xvalout = fake_amplitude.detach()
            xvalout = xvalout.cpu()
            ims = xvalout.detach().numpy()
            ims = ims[0, 0, :, :]
            qinit = (ims * 255).astype(np.uint8)
            skimage.io.imsave(save_path + '/save/' + str(epoch) + '重建.png', qinit)

            xvalout = consistency_amplitude.detach()
            xvalout = xvalout.cpu()
            ims = xvalout.detach().numpy()
            ims = ims[0, 0, :, :]
            qinit = (ims * 255).astype(np.uint8)
            skimage.io.imsave(save_path + '/save/' + str(epoch) + '相关重建.png', qinit)

            xvalout = consistency_diffraction.detach()
            xvalout = xvalout.cpu()
            ims = xvalout.detach().numpy()
            ims = ims[0, 0, :, :]
            qinit = (ims * 255).astype(np.uint8)
            skimage.io.imsave(save_path + '/save/' + str(epoch) + '相关衍射.png', qinit)

            xvalout = fake_diffraction.detach()
            xvalout = xvalout.cpu()
            ims = xvalout.detach().numpy()
            ims = ims[0, 0, :, :]
            qinit = (ims * 255).astype(np.uint8)
            skimage.io.imsave(save_path + '/save/' + str(epoch) + '衍射.png', qinit)

            xvalout = diffraction.detach()
            xvalout = xvalout.cpu()
            ims = xvalout.detach().numpy()
            ims = ims[0, 0, :, :]
            qinit = (ims * 255).astype(np.uint8)
            skimage.io.imsave(save_path + '/save/' + str(epoch) + '拍摄图.png', qinit)

            xvalout = real_amplitude.detach()
            xvalout = xvalout.cpu()
            ims = xvalout.detach().numpy()
            ims = ims[0, 0, :, :]
            qinit = (ims * 255).astype(np.uint8)
            skimage.io.imsave(save_path + '/save/' + str(epoch) + '真值图.png', qinit)

            xvalout = fake_diffraction_prepro.detach()
            xvalout = xvalout.cpu()
            ims = xvalout.detach().numpy()
            ims = ims[0, 0, :, :]
            qinit = (ims * 255).astype(np.uint8)
            skimage.io.imsave(save_path + '/save/' + str(epoch) + 'bottom_wiener.png', qinit)
    lr_scheduler_G.step()
    lr_scheduler_D.step()

    make_path(os.path.join(save_path, 'generated_10d'))

    # path for saving result
    p = os.path.join(save_path, 'generated_10d', 'iterations_' + str(epoch))
    make_path(p)
    save_data = {'iteration': epoch,
                 'Field_G_state_dict': Field_G.state_dict(),
                 'diffraction_G_state_dict': diffraction_G.state_dict(),
                 'Field_D_state_dict': Field_D.state_dict(),
                 'diffraction_D_state_dict': diffraction_D.state_dict(),
                 'distance_G_state_dict': distance_G.state_dict(),
                 'op_G': op_G.state_dict(),
                 'op_D': op_D.state_dict(),
                 'args': args}
    torch.save(save_data, os.path.join(p, "model.pth"))


