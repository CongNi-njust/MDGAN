import torch
import matplotlib.pyplot as plt
from math import pi
# rand_distance = 25*(torch.rand(size=(14, 1, 1, 1)).float()-0.5)
# print(rand_distance)
def generate_psf(distance):
    dm = 0.0038
    Nm = 2048
    r = 0.25
    d = 2.57
    f_max = 1 / dm
    du = f_max / Nm
    distance = distance*550+100
    bi = pi / r ** 2

    xm, ym = torch.meshgrid(torch.arange(-Nm * dm / 2, Nm * dm / 2, dm),
                            torch.arange(-Nm * dm / 2, Nm * dm / 2, dm), indexing='ij')
    xm = xm.unsqueeze(0).unsqueeze(0).cuda()
    ym = ym.unsqueeze(0).unsqueeze(0).cuda()

    u, v = torch.meshgrid(torch.arange(-Nm/2 * du, Nm/2 * du, du),
                          torch.arange(-Nm/2 * du, Nm/2 * du, du), indexing='ij')
    u = u.unsqueeze(0).unsqueeze(0).cuda()
    v = v.unsqueeze(0).unsqueeze(0).cuda()

    mask = 0.5 + 0.5 * torch.sign(torch.cos((bi * (xm ** 2 + ym ** 2) - pi/2)))
    # mask = torch.where(xm ** 2 + ym ** 2 > 3.5, 0, mask)

    lambda_vals = torch.arange(440, 710, 100) * 1e-6
    # wave = torch.tensor([500]).cuda()
    wave = lambda_vals.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).cuda()

    sphere = torch.exp(1j * 2 * pi * torch.sqrt(xm ** 2 + ym ** 2 + distance ** 2) / wave) / (1j * wave * distance)
    U = mask * sphere
    FU = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(U, dim=(-2, -1))), dim=(-2, -1))
    Ha = torch.exp(1j * 2 * pi * d * torch.sqrt(1 / wave ** 2 - u ** 2 - v ** 2))
    U1 = torch.fft.fftshift(torch.fft.ifft2(torch.fft.fftshift(FU * Ha, dim=(-2, -1))), dim=(-2, -1))

    Im = (U1 * torch.conj(U1)).real
    Im = torch.sum(Im, 1, keepdim=True)

    Im = Im[:, :, Nm//4:Nm*3//4, Nm//4:Nm*3//4]
    max = torch.max((torch.max(Im, -1, keepdim=True)[0]), -2, keepdim=True)[0]
    min = torch.min((torch.min(Im, -1, keepdim=True)[0]), -2, keepdim=True)[0]
    I = (Im - min) / (max - min)

    return I

# d = torch.tensor([100, 200, 400, 500,600,1000]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
# a = generate_psf(d)
#
# d1 = torch.tensor([500]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).cuda()
# a1 = generate_psf(d1)
#
# plt.figure(1)
# plt.imshow((a[0][0]-a1[0][0]).cpu().detach().numpy(), cmap='gray')
