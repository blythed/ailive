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


def _make_glances(experiment, fake, noise, content, generator, n_components):
    vutils.save_image(
        fake * 0.5 + 0.5,
        f'models/{experiment}/texture.jpg',
        normalize=False
    )
    n2 = noise[:4].repeat(1, 1, 3, 3)
    n2 = set_noise(n2, audio=content[:4], nGL=n_components)
    generator.eval()
    with torch.no_grad():
        fake2 = generator(n2)
    generator.train()

    vutils.save_image(
        fake2 * 0.5 + 0.5,
        f'models/{experiment}/texture_2.jpg',
        normalize=False,
    )


def train(
    discriminator,
    generator,
    experiment,
    texture_data,
    audio_data,
    manual_seed: typing.Optional[int] = None,
    image_size: int = 160,
    lr: float = 0.0002,
    beta1: float = 0.5,
    beta2: float = 0.999,
    n_epochs: int = 100,
    batch_size: int = 8,
    n_components: int = 20,
):
    """
    Train DC gan.

    :param discriminator: discriminator network.
    :param generator: generator network.
    :param experiment: experiment directory.
    :param texture_data: texture-dataset instance.
    :param audio_data: audio-data instance.
    :param manual_seed: manual seed for reproducibility
    :param image_size: size of image-crops
    :param lr: learning rate
    :param beta1: beta1 for optimizer
    :param beta2: beta2 for optimizer
    :param n_epochs: number of epochs to train
    :param batch_size: batch-size
    :param n_components: number of projection components for dimensionality reduction
    """

    random.seed(int(time.time() if manual_seed is None else manual_seed))
    torch.manual_seed(int(time.time()) if manual_seed is None else manual_seed)

    # torch.nn.Module.apply applies function to every recursive submodule
    discriminator.apply(_weights_init)
    generator.apply(_weights_init)
    discriminator.to(DEVICE)
    generator.to(DEVICE)

    optimizerD = optim.Adam(
        discriminator.parameters(),
        lr=lr,
        betas=(beta1, beta2),
    )
    optimizerG = optim.Adam(
        generator.parameters(),
        lr=lr,
        betas=(beta1, beta2),
    )
    texture_dataloader = torch.utils.data.DataLoader(
        texture_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    criterion = torch.nn.BCELoss()
    real_label = 1
    fake_label = 0
    size_parameter = image_size // 2 ** generator.network_depth
    noise = torch.FloatTensor(batch_size, generator.in_channels, size_parameter, size_parameter).to(DEVICE)
    noise = set_noise(noise, nGL=n_components)

    it = 0
    for epoch in range(n_epochs):
        for batch in texture_dataloader:
            content = audio_data.next().to(DEVICE)
            textures, _ = batch
            textures = textures.to(DEVICE)
            noise = set_noise(noise, audio=content, nGL=n_components)

            # take discriminator step
            discriminator.zero_grad()
            output = discriminator(textures)
            errD_real = criterion(output, output.detach() * 0 + real_label)
            errD_real.backward()
            D_x = output.mean().item()
            with torch.no_grad():
                fake = generator(noise)
            output = discriminator(fake.detach())
            errD_fake = criterion(output, output.detach() * 0 + fake_label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            optimizerD.step()

            # take generator step
            optimizerG.zero_grad()
            content = audio_data.next().to(DEVICE)
            noise = set_noise(noise, audio=content, nGL=n_components)
            fake = generator(noise)
            output = discriminator(fake)
            errG = criterion(output, output.detach() * 0 + real_label)
            errG.backward()
            optimizerG.step()

            # bookkeeping
            it += 1
            if it % 5 == 0:
                print(f'TRAIN iteration: {it}; epoch: {epoch}; d_loss: {D_x}; g_loss: {D_G_z1};')

        _make_glances(experiment, fake, noise, content, generator, n_components)
        torch.save(generator.state_dict(), f'models/{experiment}/model.pt')
