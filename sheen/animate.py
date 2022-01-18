from typing import Optional, Tuple

import numpy
import queue
import sounddevice
import tkinter
import torch

from sheen.utils import AudioTransform
from sheen.models import set_noise
from sheen import DEVICE


class Animator:
    def __init__(self,
                 netG,
                 projection,
                 audio_device: int,
                 channels: Tuple[int] = (1,),
                 L: int = 60,
                 sensitivity: float = 1.0,
                 normalizer: float = 0.0,
                 random_walk: bool = False,
                 walk_steps: int = 100,
                 n_steps: int = 0,
                 n_channels: int = 2,
                 sample_rate: Optional[int] = None,
                 downsample: int = 1,
                 window: int = 200,
                 n_smooth: int = 0,
                 pca_window: int = 2000):

        self.netG = netG
        self.projection = projection
        self.audio_device = audio_device
        self.channels = channels
        self.L = L
        self.sensitivity = sensitivity
        self.normalizer = normalizer
        self.random_walk = random_walk
        self.walk_steps = walk_steps
        self.n_steps = n_steps
        self.n_channels = n_channels
        self.sample_rate = sample_rate if sample_rate is None else self._get_sample_rate()
        self.downsample = downsample
        self.window = window
        self.n_smooth = n_smooth
        self.pca_window = pca_window

        self.projection_buffer = []

        length = int(self.window * self.sample_rate / (1000 * self.downsample))
        self.plotdata = numpy.zeros((length, n_channels))

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

        self.noise = self._init_noise()
        self.new_noise = self._init_noise()
        self.random_walk = random_walk
        self.walk_steps = walk_steps
        self.n_steps = n_steps

        self.order = self._init_order()
        self.L = L
        self.LE = self.noise.shape[1] - self.netG.nperiodic
        self.drift = torch.zeros(*[
            self.noise.shape[0],
            self.LE - self.L,
            *self.noise.shape[2:]
        ]).to(DEVICE)

        self.fDrift = 0.0
        self.audio_transform = AudioTransform()
        self.the_queue = queue.Queue()
        self.mapping = [0]  # not sure what this is.
        self.smoothing_window = 1

    def _get_sample_rate(self):
        device_info = sounddevice.query_devices(self.audio_device, 'input')
        return device_info['default_samplerate']

    def _update_projection(self, x):
        if len(self.projection_buffer) >= 80:
            self.projection_buffer = self.projection_buffer[-79:]
        self.projection_buffer.append(x[:, 0])
        if len(self.projection_buffer) > 50:
            self.projection.partial_fit(numpy.array(self.projection_buffer))
            self.projection.n_samples_seen_ = min(
                self.projection.n_samples_seen_,
                self.pca_window,
            )
            if self.n_smooth:
                xsmooth = []
                for i in range(x.shape[1]):
                    j = x.shape[1] - i
                    buf = self.projection_buffer[-self.audio_cf.n_smooth - j:-j]
                    buf = numpy.array(buf).T
                    xsmooth.append(buf.mean(1))
                xsmooth = numpy.array(xsmooth)
                self.projection.transform(xsmooth).n_time_points
            else:
                self.projection.transform(x.n_time_points).n_time_points
        else:
            x = x[:self.netG.nGL]
        return x

    def _init_noise(self):
        noise = torch.FloatTensor(
            1, self.netG.nz, self.netG.Hpx, self.netG.Wpx
        )
        noise.uniform_(-1, 1)
        noise = noise.to(DEVICE)
        return noise

    def _init_order(self):
        return torch.arange(self.L).to(DEVICE).long()

    def infer_step(self, x, noise):
        x = x[:, -self.smoothing_window:]
        x = self.normalizer * x / (x.sum(0) + 0.0001) + (1 - self.normalizer) * x
        x = self.sensitivity * x
        x = self._update_projection(x)

        x = torch.FloatTensor(x).to(DEVICE).view(1, self.L, -1, 1)

        if x.shape[2] != self.netG.Hpx:
            x = x.permute(0, 1, 3, 2)
            x = torch.nn.functional.upsample(
                x,
                size=(self.netG.Hpx, self.netG.Wpx),
                mode="bilinear"
            )

        x = x[:, self.order]

        noise = set_noise(
            noise,
            audio=x,
            nGL=self.netG.nGL,
            supressR=True,
        )

        noise[:, self.L:self.LE] = \
            torch.nn.functional.hardtanh(noise[:, self.L:self.LE], -1, 1)

        with torch.no_grad():
            img = self.netG(noise)

        img = img * 0.5 + 0.5
        img = img[0].permute(1, 2, 0)

        return img.cpu().numpy()

    def _audio_callback(self, indata, *args):
        self.the_queue.put(indata[::self.downsample, self.mapping])

    def get_noise(self):
        alpha = (self.n_steps % self.walk_steps) / self.walk_steps
        noise = alpha * self.new_noise + (1 - alpha) * self.noise
        if self.random_walk:
            self.n_steps += 1
            if (self.n_steps % self.walk_steps < 1):
                self.noise = self.new_noise
                self.new_noise = self._init_noise()
        return noise

    def create_sample(self, x):
        magnitude = self.audio_transform.get_spectrogram(x)
        magnitude = self.audio_transform.psp3(magnitude)
        magnitude = self.audio_transform.shrinkMel(magnitude)
        noise = self.get_noise()
        image = self.infer_step(magnitude, noise)
        return image

    def update(self):
        while True:
            try:
                data = self.the_queue.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = numpy.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data
        return self.create_sample(self.plotdata)

    def __iter__(self):
        with self.stream:
            while True:
                yield self.update()
