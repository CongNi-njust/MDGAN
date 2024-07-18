import torch
from torch import nn
# from model.Initialization_model import weights_initialize_xavier_normal as weights_initialize
# from functions.filtering import Laplace_op, Gauss_filt
from functions.filtering import Laplace_op, Gauss_filt
from Initialization_experiment import parse_args
def normalization(I):
    max = torch.max((torch.max(I, -1, keepdim=True)[0]), -2, keepdim=True)[0]
    min = torch.min((torch.min(I, -1, keepdim=True)[0]), -2, keepdim=True)[0]
    out = (I - min) / (max - min)
    return out
def Imaging_fft(amplitude, psf):
    batch_a, c_a, Sh_a, Sw_a = amplitude.shape
    batch_p, c_p, Sh_p, Sw_p = psf.shape
    amplitude = pad(amplitude, pad=((Sh_p-Sh_a) // 2, (Sh_p-Sh_a) // 2, (Sw_p-Sw_a) // 2, (Sw_p-Sw_a) // 2), mode="constant")
    FU = torch.fft.fft2(amplitude, dim=(-2, -1))
    PSF = torch.fft.fft2(psf, dim=(-2, -1))
    U1 = torch.fft.fftshift(torch.fft.ifft2(torch.mul(FU, PSF), dim=(-2, -1)), dim=(-2, -1)).real
    # Im = torch.pow(torch.abs(U1), 2)
    Im = normalization(U1)
    return Im
# def get_wiener_matrix(psf, gamma):
#     H = torch.fft.fft2(psf, dim=(-2, -1))
#     Habsq = torch.mul(H, torch.conj(H))
#     W = torch.div(torch.conj(H), (gamma + Habsq))
#     wiener_mat = torch.fft.ifft2(W, dim=(-2, -1))
#     return wiener_mat

# def fft_conv2d(img, w):
#     Im = torch.fft.fft2(img, dim=(-2, -1))  # img.shape=(4,1,1024,1024)
#     W = torch.fft.fft2(w, dim=(-2, -1))
#     out = torch.fft.ifft2(torch.mul(W, Im), dim=(-2, -1))
#     out = torch.pow(torch.abs(out), 2)
#     max = torch.max((torch.max(out, -1, keepdim=True)[0]), -2, keepdim=True)[0]
#     min = torch.min((torch.min(out, -1, keepdim=True)[0]), -2, keepdim=True)[0]
#     I = (out - min) / (max - min)
#     return I

class Distance_Generator(nn.Module):
    '''
    The object-to-sensor distance generator, G_psi.
    A Single diffraction pattern intensity is used as input for the network.
    '''

    def __init__(self, args):
        super(Distance_Generator, self).__init__()
        self.input_channel = 1
        self.output_channel = 1
        self.lrelu_use = args.lrelu_use
        self.use_norm = args.norm_use
        self.batch_mode = args.batch_mode

        c1 = args.initial_channel
        c2 = c1*2
        c3 = c2*2

        # Stage 1
        self.l10 = CBR(in_channel=self.input_channel, out_channel=c1, use_norm=False, kernel=7, padding=3,
                       lrelu_use=self.lrelu_use)
        self.l11 = CBR(in_channel=c1, out_channel=c1, kernel=7, padding=3,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        # Stage 2
        self.l20 = CBR(in_channel=c1, out_channel=c2, kernel=5, padding=2,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l21 = CBR(in_channel=c2, out_channel=c2, kernel=5, padding=2,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.l30 = CBR(in_channel=c2, out_channel=c2, kernel=5, padding=2,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l31 = CBR(in_channel=c2, out_channel=c2, kernel=5, padding=2,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        # Stage 3
        self.l40 = CBR(in_channel=c2, out_channel=c3, kernel=3, padding=1,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l41 = CBR(in_channel=c3, out_channel=c3, kernel=3, padding=1,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.l50 = CBR(in_channel=c3, out_channel=c3, kernel=3, padding=1,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l51 = CBR(in_channel=c3, out_channel=c3, kernel=3, padding=1,
                       use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        # output
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_out_d = nn.Conv2d(in_channels=c3, out_channels=1, kernel_size=1)
        self.out = nn.Sigmoid()

        self.mpool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # self.apply(weights_initialize)

    def forward(self, x):

        l1 = self.mpool0(self.l11(self.l10(x)))
        l2 = self.mpool0(self.l21(self.l20(l1)))
        l3 = self.mpool0(self.l31(self.l30(l2)))
        l4 = self.mpool0(self.l41(self.l40(l3)))
        l5 = self.mpool0(self.l51(self.l50(l4)))

        out_d = self.global_avg_pool(l5)
        out_d = self.out(self.conv_out_d(out_d))
        # out_d = (self.conv_out_d(out_d))
        # out_d = torch.clamp(out_d, 0, 1)

        return out_d

class Field_Generator(nn.Module):
    '''
    Complex amplitude map generator, G_theta.
    input_channel=1 for a single diffraction pattern intensity which used as an input.
    To build diffraction pattern intensity generator, input_channel=2 for input complex amplitude map and output_channel=1 for output diffraction intensity.
    '''

    def __init__(self, args, input_channel=1):
        super(Field_Generator, self).__init__()

        self.use_norm = args.norm_use
        self.input_channel = input_channel
        self.output_channel = args.output_channel
        self.lrelu_use = args.lrelu_use
        self.batch_mode=args.batch_mode

        c1 = args.initial_channel
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2
        c5 = c4*2

        self.l10 = CBR(in_channel=self.input_channel, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use)
        self.l11 = CBR(in_channel=c1, out_channel=c1, use_norm=False, lrelu_use=self.lrelu_use)
        self.SE1 = SELayer(channel=c1)

        self.l20 = CBR(in_channel=c1, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l21 = CBR(in_channel=c2, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.SE2 = SELayer(channel=c2)

        self.l30 = CBR(in_channel=c2, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l31 = CBR(in_channel=c3, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.SE3 = SELayer(channel=c3)

        self.l40 = CBR(in_channel=c3, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l41 = CBR(in_channel=c4, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.SE4 = SELayer(channel=c4)

        self.l50 = CBR(in_channel=c4, out_channel=c5, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l51 = CBR(in_channel=c5, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T5 = nn.ConvTranspose2d(in_channels=c4, out_channels=c4, kernel_size=(2,2), stride=(2,2), padding=(0,0))

        self.l61 = CBR(in_channel=c5, out_channel=c4, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l60 = CBR(in_channel=c4, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T6 = nn.ConvTranspose2d(in_channels=c3, out_channels=c3, kernel_size=(2,2), stride=(2,2), padding=(0,0))

        self.l71 = CBR(in_channel=c4, out_channel=c3, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l70 = CBR(in_channel=c3, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T7 = nn.ConvTranspose2d(in_channels=c2, out_channels=c2, kernel_size=(2,2), stride=(2,2), padding=(0,0))

        self.l81 = CBR(in_channel=c3, out_channel=c2, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l80 = CBR(in_channel=c2, out_channel=c1, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.conv_T8 = nn.ConvTranspose2d(in_channels=c1, out_channels=c1, kernel_size=(2,2), stride=(2,2), padding=(0,0))

        self.l91 = CBR(in_channel=c2, out_channel=c1, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)
        self.l90 = CBR(in_channel=c1, out_channel=c1, use_norm=self.use_norm, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        if self.output_channel == 2:
            self.conv_out_amplitdue = nn.Conv2d(in_channels=c1, out_channels=1, kernel_size=(1, 1), padding=0)
            self.conv_out_phase = nn.Conv2d(in_channels=c1, out_channels=1, kernel_size=(1, 1), padding=0)
            self.SE_out_amplitude = SELayer(channel=c1)
            self.SE_out_phase = SELayer(channel=c1)

        else:
            self.conv_out_holo = nn.Conv2d(in_channels=c1, out_channels=1, kernel_size=(1, 1), padding=0)
            self.SE_out_holo = SELayer(channel=c1)

        # self.apply(weights_initialize)
        self.mpool0 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):

        l1 = self.l11(self.l10(x))
        l1_pool = self.mpool0(l1)

        l2 = self.l21(self.l20(l1_pool))
        l2_pool = self.mpool0(l2)

        l3 = self.l31(self.l30(l2_pool))
        l3_pool = self.mpool0(l3)

        l4 = self.l41(self.l40(l3_pool))
        l4_pool = self.mpool0(l4)

        l5 = self.conv_T5(self.l51(self.l50(l4_pool)))

        l6 = torch.cat([l5, self.SE4(l4)], dim=1)
        l6 = self.conv_T6(self.l60(self.l61(l6)))

        l7 = torch.cat([l6, self.SE3(l3)], dim=1)
        l7 = self.conv_T7(self.l70(self.l71(l7)))

        l8 = torch.cat([l7, self.SE2(l2)], dim=1)
        l8 = self.conv_T8(self.l80(self.l81(l8)))

        l9 = torch.cat([l8, self.SE1(l1)], dim=1)
        out = self.l90(self.l91(l9))

        # l8 = torch.cat([l7, self.SE2(l2)], dim=1)
        # out = self.l80(self.l81(l8))
        out = self.conv_out_holo(self.SE_out_holo(out))

        return out

class SELayer(nn.Module):
    '''
    Squeeze-and-excitation network used in G_theta and D_eta.
    '''
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Discriminator(nn.Module):
    '''
    Complex amplitude map discriminator, D_eta.
    input_channel=3 for original image, high pass filtered image, and low pass filtered image.
    output_channle=1 to generate whether the input image is real or fake.
    '''
    def __init__(self, args, input_channel=1):
        super(Discriminator, self).__init__()
        self.input_channel = input_channel
        self.output_channel = 1
        self.use_norm = args.norm_use
        self.lrelu_use = args.lrelu_use
        self.batch_mode=args.batch_mode
        self.lrelu_slope = args.lrelu_slope

        c1 = args.initial_channel
        c2 = c1*2
        c3 = c2*2
        c4 = c3*2

        self.l_r = CBR(in_channel=self.input_channel, out_channel=c1, use_norm=False, kernel=4, padding=0,
                       stride=2, lrelu_use=self.lrelu_use, batch_mode=self.batch_mode)

        self.SE_rl = SELayer(channel=c1)

        self.l20 = CBR(in_channel=c1, out_channel=c2, use_norm=self.use_norm, kernel=4, padding=0, stride=2,
                       lrelu_use=self.lrelu_use, slope=self.lrelu_slope, batch_mode=self.batch_mode)
        self.l30 = CBR(in_channel=c2, out_channel=c3, use_norm=self.use_norm, kernel=4, padding=0, stride=2,
                       lrelu_use=self.lrelu_use, slope=self.lrelu_slope, batch_mode=self.batch_mode)
        self.l40 = CBR(in_channel=c3, out_channel=c4, use_norm=self.use_norm, kernel=4, padding=0, stride=1,
                       lrelu_use=self.lrelu_use, slope=self.lrelu_slope, batch_mode=self.batch_mode)

        self.conv_out = nn.Conv2d(in_channels=c4, out_channels=self.output_channel, kernel_size=(1, 1), stride=(1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.laplace_op = Laplace_op(args)
        self.gauss_op = Gauss_filt()

        # self.apply(weights_initialize)

    def forward(self, field):

        l10_r = self.l_r(field)
        l0 = self.SE_rl(l10_r)
        l1 = self.l20(l0)
        l2 = self.l30(l1)
        l3 = self.l40(l2)

        out = self.conv_out(self.avg_pool(l3))
        return out




class CBR(nn.Module):
    '''
    Convolution-norm-leaky_relu block.
    batch_mode:
    'I': Instance normalization
    'B': Batch normalization
    'G': Group normalization
    lrelu_use: defalut is True. If False, ReLU is used.
    Other parameters: used for 2D-convolution layer.
    '''

    def __init__(self, in_channel, out_channel, padding=1, use_norm=True, kernel=3, stride=1
                 , lrelu_use=False, slope=0.1, batch_mode='G', rate=1):
        super(CBR, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_norm = use_norm
        self.lrelu = lrelu_use

        self.Conv = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=(kernel, kernel), stride=(stride, stride),
                                  padding=padding, dilation=(rate, rate))

        if self.use_norm:
            if batch_mode == 'I':
                self.Batch = nn.InstanceNorm2d(self.out_channel)
            elif batch_mode == 'G':
                self.Batch = nn.GroupNorm(self.out_channel//16, self.out_channel)
            else:
                self.Batch = nn.BatchNorm2d(self.out_channel)

        self.lrelu = nn.LeakyReLU(negative_slope=slope)
        self.relu = nn.ReLU()


    def forward(self, x):

        if not self.lrelu:
            out = self.relu(self.Batch(self.Conv(x)))

        else:
            if self.use_norm:
                out = self.lrelu(self.Batch(self.Conv(x)))
            else:
                out = self.lrelu(self.Conv(x))

        return out

class WienerBlock(nn.Module):
    def __init__(self, args):
        super(WienerBlock, self).__init__()
        self.gamma = args.wiener_gamma

    def forward(self, img, psf):
        #主函数
        psf = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(psf, dim=(-2, -1))), dim=(-2, -1))
        Habsq = torch.mul(psf, torch.conj(psf))
        W = torch.div(torch.conj(psf), (self.gamma + Habsq))
        I = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(img, dim=(-2, -1))), dim=(-2, -1))
        IW = torch.mul(I, W)
        out = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(IW, dim=(-2, -1))), dim=(-2, -1)).real
        I = normalization(out[:, :, 672:928, 672:928])
        # out = self.Batch(out)
        return I.float()