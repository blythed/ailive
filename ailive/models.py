import numpy as np
import torch
import torch.nn as nn

norma = nn.BatchNorm2d
from ailive import DEVICE


def init_wave(nPeriodic, nGL, Kperiodic=50):
            buf = []  # [0,0.5,0.5,0,1,0,0,1]
            for  i in range(nPeriodic // 4):
                v = 0.5 + i / float(nPeriodic / 2)  # #so 0.5 to 1
                buf += [0, v, v, 0]
                buf += [0, -v, -v, 0]  # so from other quadrants as well..
            awave = np.array(buf, dtype=np.float32) * np.pi
            print ("awave", awave)
            # some noise
            awave += np.random.randn(awave.shape[0]) * 0.1
            # for 2 spatial dims at the end and then for batch dim in front
            awave = torch.FloatTensor(awave).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            if Kperiodic == 0:
                return nn.Parameter(awave)
            else:
                lin1 = nn.Conv2d(nGL, Kperiodic, 1, bias=True)
                rel = nn.ReLU(True)
                lin2 = nn.Conv2d(Kperiodic, nPeriodic * 2 , 1, bias=True)
                return nn.Parameter(awave) , nn.Sequential(*[lin1, rel, lin2])


def set_wave(noise, nPeriodic, net, nGL, offset=None, Kperiodic=50):
    if offset is None:
        offset = (noise[:, :nPeriodic, :1, :1] * 1.0).uniform_(-1, 1) * 6.28

    offset = offset.repeat(1, 1, noise.shape[2], noise.shape[3])
    if not net.training:
        offset *= 0

    if Kperiodic == 0:
        raw = net.waver * noise[:, -2 * nPeriodic:]
    else:
        raw = (net.wavenet.forward(noise[:, :nGL]) * 1 + net.waver) * noise[:, -2 * nPeriodic:]
    wave = torch.sin(raw[:, ::2] + raw[:, 1::2] + offset)
    return torch.cat([noise[:, :-2 * nPeriodic], wave], 1)


def set_noise(noise,
              audio=None,
              nGL=60,
              nPeriodic=0,
              nz=100,
              supressR=False):
    """
    :param noise:
    :param audio:
    :param nGL:
    :param nPeriodic:
    :param nz:
    :param supressR:
    """
    noise = 1.0 * noise
    if not supressR:
        noise.uniform_(-1, 1)
    assert(nGL > 0)
    if nGL:
        noise[:, :nGL] = noise[:, :nGL, :1, :1].repeat(
            1, 1, noise.shape[2], noise.shape[3]
        )
        if audio is not None:
            if len(audio.shape) == 2:
                audio = audio.unsqueeze(-1).unsqueeze(-1)
            noise[:, :nGL] = audio
    if nPeriodic:
        xv, yv = np.meshgrid(np.arange(noise.shape[2]),
                             np.arange(noise.shape[3]), indexing='ij')
        c = torch.FloatTensor(
            np.concatenate([xv[np.newaxis], yv[np.newaxis]], 0)[np.newaxis]
        )
        if c.data.type() != noise.data.type():
            c = c.to(DEVICE)
        c = c.repeat(noise.shape[0], nPeriodic, 1, 1)
        out = torch.cat([noise[:, :nz - nPeriodic], c], 1)
        return out
    return noise


class Discriminator(nn.Module):
    def __init__(self, ndf, nDep, nc=3, Ctype=1, bOut1=True):
        super().__init__()
        layers = []
        of = nc
        for i in range(nDep):
            if i == nDep - 1 and bOut1:
                nf = 1
            else:
                nf = ndf * 2 ** i
            if Ctype == 1:
                layers += [nn.Conv2d(of, nf, 5, 2, 2)]
            else:
                layers += [nn.Conv2d(of, nf, 4, 2, 1)]

            if i != 0 and i != nDep - 1:
                layers += [norma(nf)]

            if i < nDep - 1:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [nn.Sigmoid()]
            of = nf
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output[:, :, 1:-1, 1:-1]


class Generator(nn.Module):
    def __init__(self, ngf, nDep, nz, nGL, Ctype=1, nPeriodic=0, nc=3,
                 Kperiodic=50):
        """
        :param ngf:
        :param nDep:
        :param nz:
        :param Ctype:
        :param nPeriodic:
        :param nc:
        :param Kperiodic:
        """

        super().__init__()
        self.nPeriodic = nPeriodic
        if self.nPeriodic:
            if Kperiodic == 0:
                self.waver = init_wave(nPeriodic, nGL, Kperiodic=Kperiodic)
            else:
                self.waver, self.wavenet = init_wave(nPeriodic, nGL, Kperiodic=Kperiodic)

        self.nGL = nGL

        layers = []
        of = nz
        for i in range(nDep):
            if i == nDep - 1:
                nf = nc
            else:
                nf = ngf * 2 ** (nDep - 2 - i)
            if Ctype == 1:
                layers += [nn.Upsample(scale_factor=2, mode='nearest')]
                layers += [nn.Conv2d(of, nf, 4 + 1, 1, 2)]
            else:
                layers += [nn.ConvTranspose2d(of, nf, 6, 2, 2)]
            if i == nDep - 1:
                layers += [nn.Tanh()]
            else:
                layers += [norma(nf)]
                layers += [nn.ReLU(True)]
            of = nf
        self.main = nn.Sequential(*layers)

    def forward(self, input, offset=None):
        if self.nPeriodic:
            input = set_wave(input, self.nPeriodic, self, self.nGL, offset=offset)
        output = self.main(input)
        return output
