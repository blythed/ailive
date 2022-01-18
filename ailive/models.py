import torch


def set_noise(noise,
              audio=None,
              nGL=60,
              supressR=False):

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
    return noise


class Discriminator(torch.nn.Module):
    def __init__(self, ndf, nDep, nc=3, Ctype=1, bOut1=True):
        super().__init__()
        layers = []
        for i in range(nDep):
            if i == nDep - 1 and bOut1:
                nf = 1
            else:
                nf = ndf * 2 ** i
            if Ctype == 1:
                layers += [torch.nn.Conv2d(nc, nf, 5, 2, 2)]
            else:
                layers += [torch.nn.Conv2d(nc, nf, 4, 2, 1)]

            if i != 0 and i != nDep - 1:
                layers += [torch.nn.BatchNorm2d(nf)]

            if i < nDep - 1:
                layers += [torch.nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [torch.nn.Sigmoid()]
            nc = nf
        self.main = torch.nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output[:, :, 1:-1, 1:-1]


class Generator(torch.nn.Module):
    def __init__(self, ngf, nDep, nz, nGL, Ctype=1, nc=3):

        super().__init__()
        self.nGL = nGL

        layers = []
        of = nz
        for i in range(nDep):
            if i == nDep - 1:
                nf = nc
            else:
                nf = ngf * 2 ** (nDep - 2 - i)
            if Ctype == 1:
                layers += [torch.nn.Upsample(scale_factor=2, mode='nearest')]
                layers += [torch.nn.Conv2d(of, nf, 4 + 1, 1, 2)]
            else:
                layers += [torch.nn.ConvTranspose2d(of, nf, 6, 2, 2)]
            if i == nDep - 1:
                layers += [torch.nn.Tanh()]
            else:
                layers += [torch.nn.BatchNorm2d(nf)]
                layers += [torch.nn.ReLU(True)]
            of = nf
        self.main = torch.nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output
