from typing import Optional, Tuple

import numpy
import queue
import sounddevice
import tkinter
import torch

import sheen.projections
from sheen.audio_utils import AudioTransform
from sheen.models import set_noise
from sheen import DEVICE


class Animator:
    def __init__(self,
                 generator: torch.nn.Module,
                 projection: sheen.projections.Projection,
                 audio_device: int,
                 channels: Tuple[int] = (1,),
                 sensitivity: float = 1.0,
                 normalizer: float = 0.0,
                 sample_rate: Optional[int] = None,
                 down_sample: int = 1,
                 window: int = 200,
                 noise_width: int = 15):
        """
        Animator class to generate image samples based on live streaming audio.

        :param generator: generating neural network
        :param projection: projection mapping raw audio mel spectrograms to component space
        :param audio_device: number of audio device (do `sheen list_devices` to find out)
        :param channels: list of channels (no idea why)
        :param sensitivity: sensitivity to audio
        :param normalizer: degree to which the audio is normalized
        :param sample_rate: sample rate of audio (auto-determined if not provided)
        :param down_sample: down-sampling factor
        :param window: number of samples to use in calculating latest spectrogram
        :param noise_width: size of noise tensor which is input into generator
        """

        self.generator = generator
        self.projection = projection
        self.audio_device = audio_device
        self.channels = channels
        self.sensitivity = sensitivity
        self.normalizer = normalizer
        self.sample_rate = sample_rate if sample_rate is None else self._get_sample_rate()
        self.downsample = down_sample
        self.window = window
        self.noise_width = noise_width

        # these variables need to initialized but not set in __init__
        self.projection_buffer = []
        self.random_walk = False
        self.walk_steps = 100
        self.n_steps = 0
        self.audio_transform = AudioTransform()
        self.the_queue = queue.Queue()
        self.mapping = [0]  # not sure what this is.
        self.smoothing_window = 1

        # maybe this has to do with how the FFTs are created ?
        length = int(self.window * self.sample_rate / (1000 * self.downsample))  # what's this 1000?
        self.data_buffer = numpy.zeros((length, len(channels)))
        self.stream = sounddevice.InputStream(
            device=self.audio_device,
            channels=max(self.channels),
            samplerate=self.sample_rate,
            callback=self._audio_callback,
        )

        root = tkinter.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.aspect_ratio = self.screen_height / self.screen_width
        root.withdraw()

        self.noise_height = round(self.noise_width * self.aspect_ratio)
        self.noise = self._init_noise()
        self.new_noise = self._init_noise()

    def _get_sample_rate(self):
        device_info = sounddevice.query_devices(self.audio_device, 'input')
        return device_info['default_samplerate']

    def _update_projection(self, x):
        self.projection_buffer.append(x[:, 0])
        if len(self.projection_buffer) > 80:  # why 80 - what is this?
            self.projection_buffer = self.projection_buffer[-80:]  # ??
        if len(self.projection_buffer) > 50:   # why 50 - what is this?
            self.projection.update(numpy.array(self.projection_buffer))
            return self.projection.transform(x.T).T
        else:
            x = x[:self.projection.n_components]
        return x

    def _init_noise(self):
        noise = torch.FloatTensor(
            1, self.generator.in_channels, self.noise_height, self.noise_width
        )
        noise.uniform_(-1, 1)
        noise = noise.to(DEVICE)
        return noise

    def infer_step(self, x, noise):
        # doesn't this fuck everything up ??????????????????
        x = x[:, -self.smoothing_window:]  ## ?????????????????
        x = self.normalizer * x / (x.sum(0) + 0.0001) + (1 - self.normalizer) * x
        x = self.sensitivity * x
        x = self._update_projection(x)

        # why the -1 here? I think this expands the dimensions to fit the network
        x = torch.FloatTensor(x).to(DEVICE).view(1, self.projection.n_components, -1, 1)

        if x.shape[2] != self.noise_height:  # why would this ever not be the case?
            x = x.permute(0, 1, 3, 2)
            x = torch.nn.functional.upsample(
                x,
                size=(self.noise_height, self.generator.noise_width),
                mode="bilinear"
            )

        # this function is confusing - needs to be split up -
        # 1. creating noise
        # 2. interpolating new-noise with old
        # 3. adding audio to noise
        noise = set_noise(
            noise,
            audio=x,
            nGL=self.projection.n_components,
            supressR=True,
        )

        noise[:, self.projection.n_components:] = \
            torch.nn.functional.hardtanh(noise[:, self.projection.n_components:], -1, 1)

        with torch.no_grad():
            img = self.generator(noise)

        # why these transformations?
        img = img * 0.5 + 0.5
        img = img[0].permute(1, 2, 0)

        return img.cpu().numpy()

    def _audio_callback(self, indata, *args):
        self.the_queue.put(indata[::self.downsample, self.mapping])

    def _get_noise(self):
        # get how far we are along in the interpolation of one noise to another
        alpha = (self.n_steps % self.walk_steps) / self.walk_steps
        noise = alpha * self.new_noise + (1 - alpha) * self.noise
        if self.random_walk:
            self.n_steps += 1
            if (self.n_steps % self.walk_steps < 1):
                self.noise = self.new_noise
                self.new_noise = self._init_noise()
        return noise

    def _create_sample(self, x):
        magnitude = self.audio_transform(x)
        noise = self._get_noise()
        image = self.infer_step(magnitude, noise)
        return image

    def _update(self):
        while True:
            try:
                data = self.the_queue.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            # what does this roll do?
            self.data_buffer = numpy.roll(self.data_buffer, -shift, axis=0)
            self.data_buffer[-shift:, :] = data
        return self._create_sample(self.data_buffer)

    def __iter__(self):
        with self.stream:
            while True:
                yield self._update()
