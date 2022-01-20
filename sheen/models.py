import torch
from aiconfig import build


def load_model(experiment):
    """
    Load model into memory, including weights.

    :param experiment: directory inside "./models"
    """
    model = build(f'models/{experiment}/explicit.json', target='generator')[0]
    model.load_state_dict(torch.load(f'models/{experiment}/model.pt'))
    return model


def set_noise(noise,
              audio=None,
              nGL=60,
              supressR=False):  # suppress random?
    """
    Create noise tensor with the purpose of inputting to the generator neural net.
    Certain dimensions of this tensor are simply the audio repeated.
    TODO consider renaming/ redesigning.

    :param noise: noise tensor (to be filled with new-noise)
    :param audio: audio signal
    :param nGL: number of components to fill with audio
    :param supressR:
    """
    assert(nGL > 0)

    if not supressR:
        noise.uniform_(-1, 1)

    # what the f. is this parameter nGL?
    # why are we doing this?
    noise[:, :nGL] = noise[:, :nGL, :1, :1].repeat(
        1, 1, noise.shape[2], noise.shape[3]
    )
    # wtf?
    if audio is not None:
        if len(audio.shape) == 2:
            audio = audio.unsqueeze(-1).unsqueeze(-1)
        noise[:, :nGL] = audio  # there is an ellypsis here
    return noise


class Discriminator(torch.nn.Module):
    # add sane defaults
    def __init__(self, channel_number_parameter, network_depth, image_channels=3,
                 convolution_type='regular', predict_binary=True):
        """
        Discriminator neural network.

        :param channel_number_parameter: parameter governing the number of channels at a given depth
        :param network_depth: number of layers in the network
        :param image_channels: number of channels in the input image
        :param convolution_type: type of convolution {'regular', 'transpose'}
        :param predict_binary: toggle to True to predict binary output (TODO required?)
        """
        super().__init__()
        self.channel_number_parameter = channel_number_parameter
        self.network_depth = network_depth
        self.image_channels = image_channels
        self.convolution_type = convolution_type
        self.predict_binary = predict_binary

        layers = []
        for i in range(network_depth):
            if i == network_depth - 1 and predict_binary:
                out_channels = 1
            else:
                out_channels = channel_number_parameter * 2 ** i
            # is convolution type ever anything else?
            if convolution_type == 'regular':
                layers += [torch.nn.Conv2d(image_channels, out_channels, 5, 2, 2)]
            elif convolution_type == 'transpose':
                layers += [torch.nn.Conv2d(image_channels, out_channels, 4, 2, 1)]
            else:
                raise ValueError('convolution_type')

            if i != 0 and i != network_depth - 1:
                layers += [torch.nn.BatchNorm2d(out_channels)]

            if i < network_depth - 1:
                layers += [torch.nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [torch.nn.Sigmoid()]
            image_channels = out_channels
        self.main = torch.nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output[:, :, 1:-1, 1:-1]


class Generator(torch.nn.Module):
    # TODO add sane defaults
    def __init__(self, channel_number_parameter, network_depth, in_channels,
                 convolution_type='regular', image_channels=3):
        """
        Generator neural network.

        :param channel_number_parameter: parameter governing the number of channels at a given depth
        :param network_depth: number of layers in the network
        :param in_channels: number of channels in noise tensor to input
        :param convolution_type: type of convolution {'regular', 'transpose'}
        :param image_channels: number of channels in the generated image
        """

        super().__init__()
        self.channel_number_parameter = channel_number_parameter
        self.network_depth = network_depth
        self.in_channels = in_channels
        self.convolution_type = convolution_type
        self.image_channels = image_channels

        layers = []
        for i in range(network_depth):
            if i == network_depth - 1:
                out_channels = image_channels
            else:
                out_channels = channel_number_parameter * 2 ** (network_depth - 2 - i)
            if convolution_type == 'regular':
                layers += [torch.nn.Upsample(scale_factor=2, mode='nearest')]
                layers += [torch.nn.Conv2d(in_channels, out_channels, 5, 1, 2)]
            # do we ever use the transpose case?
            elif convolution_type == 'transpose':
                layers += [torch.nn.ConvTranspose2d(in_channels, out_channels, 6, 2, 2)]
            else:
                raise ValueError('convolution_type')
            if i == network_depth - 1:
                layers += [torch.nn.Tanh()]
            else:
                layers += [torch.nn.BatchNorm2d(out_channels)]
                layers += [torch.nn.ReLU(True)]
            in_channels = out_channels
        self.main = torch.nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output
