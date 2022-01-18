import random
import time
import typing

import torch
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from sheen import DEVICE
from sheen.models import set_noise


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def _make_glances(experiment, epoch, fake, noise, content, netG):
    vutils.save_image(
        fake * 0.5 + 0.5,
        f'models/{experiment}/glances/texture_{epoch:03}.jpg',
        normalize=False
    )
    n2 = noise[:4].repeat(1, 1, 3, 3)
    n2 = set_noise(n2, audio=content[:4], nGL=netG.nGL)
    netG.eval()
    with torch.no_grad():
        fake2 = netG(n2)
    netG.train()

    vutils.save_image(
        fake2 * 0.5 + 0.5,
        f'models/{experiment}/glances/texture_2_{epoch:03}.jpg',
        normalize=False,
    )


def train(
    netD,
    netG,
    experiment,
    texture_data,
    audio_data,
    manualSeed: typing.Optional[int] = None,
    imageSize: int = 160,
    lr: float = 0.0002,
    beta1: float = 0.5,
    beta2: float = 0.999,
    n_epochs: int = 100,
    batchSize: int = 8,
):

    random.seed(int(time.time() if manualSeed is None else manualSeed))
    torch.manual_seed(int(time.time()) if manualSeed is None else manualSeed)

    # torch.nn.Module.apply applies function to every recursive submodule
    netD.apply(_weights_init)
    netG.apply(_weights_init)
    netD.to(DEVICE)
    netG.to(DEVICE)

    optimizerD = optim.Adam(
        netD.parameters(),
        lr=lr,
        betas=(beta1, beta2),
    )
    optimizerG = optim.Adam(
        netG.parameters(),
        lr=lr,
        betas=(beta1, beta2),
    )
    texture_dataloader = torch.utils.data.DataLoader(
        texture_data, batch_size=batchSize,
        shuffle=True,
        num_workers=1
    )

    criterion = torch.nn.BCELoss()
    real_label = 1
    fake_label = 0
    nz_other = imageSize // 2 ** netG.nDep
    noise = torch.FloatTensor(batchSize, netG.nz, nz_other, nz_other).to(DEVICE)
    noise = set_noise(noise, nGL=netG.nGL)

    it = 0
    for epoch in range(n_epochs):
        for batch in texture_dataloader:
            content = audio_data.next().to(DEVICE)
            textures, _ = batch
            textures = textures.to(DEVICE)
            noise = set_noise(noise, audio=content, nGL=netG.nGL)

            # take discriminator step
            netD.zero_grad()
            output = netD(textures)
            errD_real = criterion(output, output.detach() * 0 + real_label)
            errD_real.backward()
            D_x = output.mean().item()
            with torch.no_grad():
                fake = netG(noise)
            output = netD(fake.detach())
            errD_fake = criterion(output, output.detach() * 0 + fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            optimizerD.step()

            # take generator step
            optimizerG.zero_grad()
            content = audio_data.next().to(DEVICE)
            noise = set_noise(noise, audio=content, nGL=netG.nGL)
            fake = netG(noise)
            output = netD(fake)
            errG = criterion(output, output.detach() * 0 + real_label)
            errG.backward()
            optimizerG.step()

            it += 1
            if it % 5 == 0:
                print(f'TRAIN iteration: {it}; epoch: {epoch}; '
                      f'd_loss: {D_x}; g_loss: {D_G_z1};')

        _make_glances(experiment, epoch, fake, noise, content, netG)
        torch.save(netG.state_dict(), f'models/{experiment}/model.pt')
