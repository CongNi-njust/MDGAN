import torch
import torch.nn as nn
from psf_generator import *
import argparse
import os
from dataload import *
from Inverse_operator import Distance_Generator, Field_Generator, Discriminator, WienerBlock, Imaging_fft, normalization
import skimage.io
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim
def make_path(path):
    import os
    if not os.path.isdir(path):
        os.mkdir(path)
def psnr(original, reconstructed):
    mse = F.mse_loss(original, reconstructed)
    psnr = 10 * torch.log10(1 / mse)
    return psnr.item()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_source_dir', default=r'D:\.../')
    parser.add_argument('--train_target_dir', default=r'D:\.../')
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--checkpoint', default=r'D:\.../s/')
    parser.add_argument("--wiener_gamma", default=1e5, type=float)
    parser.add_argument("--norm_use", default=True, type=bool)
    parser.add_argument("--lrelu_use", default=True, type=bool)
    parser.add_argument("--lrelu_slope", default=0.1, type=float)
    parser.add_argument("--batch_mode", default='B', type=str)
    parser.add_argument("--zero_padding", default=True, type=bool)
    parser.add_argument('--initial_channel', default=8, type=int)
    parser.add_argument('--output_channel', default=1, type=int)
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--lpips', default='vgg')
    return parser.parse_args()
args = parse_args()
dtype = torch.cuda.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
save_path = 'D:/nc/光学中心服务器/result_test_185/'
make_path(save_path)
average_pool = nn.AvgPool2d(2, stride=2)

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
# loss
loss_fn_lpips = lpips.LPIPS(net=args.lpips).to(device)

start_epoch = 0
checkpoint = os.path.join(args.checkpoint)
ckpt = torch.load(checkpoint + '/iterations_185/model.pth')  # 这里需要实时更改从哪里读入

distance_G.load_state_dict(ckpt['distance_G_state_dict'])
diffraction_G.load_state_dict(ckpt['diffraction_G_state_dict'])
Field_G.load_state_dict(ckpt['Field_G_state_dict'])
# Field_D.load_state_dict(ckpt['Field_D_state_dict'])
# diffraction_D.load_state_dict(ckpt['diffraction_D_state_dict'])
# Field_G.load_state_dict(ckpt['Field_G_state_dict'])
# diffraction_G.load_state_dict(ckpt['diffraction_G_state_dict'])
# distance_G.load_state_dict(ckpt['distance_G_state_dict'])
# args.load_state_dict(ckpt['args'])

# train_main
torch.no_grad()
# Field_G.eval()
# diffraction_G.eval()
# distance_G.eval()

psnr_GAN = []
lpips_GAN = []
distance_all = []
ssim_GAN = []
for i, batch in enumerate(data):
    diffraction, real_amplitude, distance = batch
    diffraction = normalization(diffraction).to(device)
    real_amplitude = normalization(average_pool(real_amplitude)).to(device)
    real_distance = ((distance - 100) / 550).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
    # rand_distance = torch.rand(size=(args.batch_size, 1, 1, 1)).to(device=device).float()
    fake_distance = distance_G(diffraction)
    # fake_distance = torch.tensor([25/60, 20/60, 30/60, 40/60, 55/60]).reshape(5, 1, 1, 1).cuda()
    psf_guess = generate_psf(fake_distance)
    diffraction_prepro = wiener(diffraction, psf_guess)
    fake_amplitude = Field_G(diffraction_prepro)[:, :, 384:384 + 256, 384:384 + 256]

    Psnr = psnr(fake_amplitude, real_amplitude)
    psnr_GAN.append(Psnr)
    Lpips = loss_fn_lpips(fake_amplitude, real_amplitude)
    lpips_GAN.append(Lpips.item())
    ssim_val = ssim(fake_amplitude, real_amplitude, data_range=1, size_average=False).item()
    ssim_GAN.append(ssim_val)

    fake_amplitude = normalization(fake_amplitude)
    Xvalout = fake_amplitude.detach()
    Xvalout = Xvalout.cpu()
    ims = Xvalout.numpy()

    ims1 = ims[0][0]

    # plt.figure(3)
    # plt.imshow(ims1, cmap='gray')
    # plt.show()
    distance_all.append((fake_distance.item())*550+100)

    # min_val = np.min(ims1)
    # max_val = np.max(ims1)

    # Qinit = (ims1 - min_val) / (max_val - min_val)
    # Qinit = (Qinit * 255).astype(np.uint8)
    Qinit = (ims1 * 255).astype(np.uint8)
    num_str = "{:05d}".format(i)
    # # psnr_str = "{:.2f}".format(Psnr)
    # # ssim_str = "{:.2f}".format(ssim_val)
    skimage.io.imsave(save_path + num_str + '.png', Qinit)
psnr_GAN = np.array(psnr_GAN)
ssim_GAN = np.array(ssim_GAN)
lpips_GAN = np.array(lpips_GAN)
distance_ll = np.array(distance_all)
np.savetxt('psnr_GAN_185.txt', psnr_GAN)
np.savetxt('ssim_GAN_185.txt', ssim_GAN)
np.savetxt('distance_185.txt', distance_ll)