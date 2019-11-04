import numpy as np
import pickle
import queue
import sounddevice as sd
from sklearn.decomposition import  IncrementalPCA
import sys
import time
import tkinter as tk
import warnings

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F

from ailive.utils import psp3, L, AA
from ailive.models import Generator, setNoise
from ailive import DEVICE


class Animator:
    def __init__(self,
                 model_cf,
                 sensitivity_cf,
                 audio_cf):

        root = tk.Tk()
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        root.withdraw()
        self.aspect_ratio = self.screen_height / self.screen_width

        self.sensitivity_cf = sensitivity_cf
        self.audio_cf = audio_cf

        self.model_cf = model_cf

        self.noise = self._init_noise()
        self.new_noise = self._init_noise()
        self.random_walk = False
        self.walk_steps = 100
        self.n_steps = 0

        self.order = self._init_order()
        self.LE = self.noise.shape[1] - self.model_cf.nperiodic
        self.drift = torch.zeros(*[
            self.noise.shape[0],
            self.LE - self.L,
            *self.noise.shape[2:]
        ]).to(DEVICE)

        self.fDrift = 0.0
        self.audioA = AA()
        self.q = queue.Queue()

        device_info = sd.query_devices(self.audio_cf.device, 'input')
        self.samplerate = device_info['default_samplerate']
        length = int(self.audio_cf.window * self.samplerate /
                     (1000 * self.audio_cf.downsample))
        self.plotdata = np.zeros((length, len(audio_cf.channels)))

        self.stream = sd.InputStream(
            device=self.audio_cf.device,
            channels=max(self.audio_cf.channels),
            samplerate=self.samplerate,
            callback=self._audio_callback,
        )

        if self.audio_cf.bPCA:
            self._init_pca()

        self.mapping = [0]

    @property
    def model_cf(self):
        return self._model_cf

    @model_cf.setter
    def model_cf(self, value):
        print(value)
        self.L = value.nGL if self.audio_cf.bPCA else L
        value.Hpx = round(value.Wpx * self.aspect_ratio)
        self._model_cf = value
        model = self._init_model()
        self.model = self._load_model(self.model_cf.path, model)

    def _init_pca(self):
        self.pcabuf = []
        self.ipca = IncrementalPCA(n_components=20, copy=False)

    def _update_pca(self, x):
        if len(self.pcabuf) >= 80:
            self.pcabuf = self.pcabuf[-79:]
        self.pcabuf.append(x[:, 0])
        if len(self.pcabuf) > 50:
            self.ipca.partial_fit(np.array(self.pcabuf))
            x = self.ipca.transform(x.T).T
            self.ipca.n_samples_seen_ = min(
                self.ipca.n_samples_seen_,
                self.audio_cf.pca_window,
            )
        else:
            x = x[:self.model_cf.nGL]
        return x

    def _update_monitor(self, *args):
        self.line.set_ydata(self.monitor_data[-50:])

    def _init_model(self, model_cf=None):
        if model_cf is None:
            model_cf = self.model_cf
        return Generator(
            model_cf.ngf,
            model_cf.nDep,
            model_cf.nz,
            model_cf.nGL,
            Ctype=model_cf.Ctype,
            nPeriodic=model_cf.nperiodic,
            Kperiodic=model_cf.Kperiodic,
        )

    def _load_model(self, model_path, model):
        model.load_state_dict(torch.load(model_path, map_location=lambda x, loc: x))
        model.eval()
        model = model.to(DEVICE)
        return model

    def _init_noise(self):
        noise = torch.FloatTensor(
            1, self.model_cf.nz, self.model_cf.Hpx, self.model_cf.Wpx
        )
        noise.uniform_(-1, 1)
        noise = noise.to(DEVICE)
        return noise

    def _init_order(self):
        return  torch.arange(self.L).to(DEVICE).long()

    def infer_step(self, x, noise):
        if self.sensitivity_cf.normalize:
            x = self.sensitivity_cf.normalize * x / (x.sum() + 0.0001)  + (1 - self.sensitivity_cf.normalize) * x
        else:
            x = self.sensitivity_cf.level * x

        if self.audio_cf.bPCA:
            x = self._update_pca(x)

        x = torch.FloatTensor(x).to(DEVICE).view(1, self.L, -1, 1)

        if x.shape[2] != self.model_cf.Hpx:
            x = x.permute(0, 1, 3, 2)
            x = F.upsample(
                x,
                size=(self.model_cf.Hpx, self.model_cf.Wpx),
                mode="bilinear"
            )

        x = x[:, self.order]

        noise = setNoise(
            noise,
            audio=x,
            nGL=self.model_cf.nGL,
            nPeriodic=self.model_cf.nperiodic,
            nz=self.model_cf.nz,
            supressR=True,
        )

        # noise[:, self.L:self.LE] += self.drift
        # self.drift[noise[:, self.L:self.LE] ** 2 > 1] *= -1 # what is being tried here?
        noise[:, self.L:self.LE] = \
            nn.functional.hardtanh(noise[:, self.L:self.LE], -1, 1)

        with torch.no_grad():
            img = self.model(noise)

        img = img * 0.5 + 0.5
        img = img[0].permute(1, 2, 0)

        return img.cpu().numpy()

    def _audio_callback(self, indata, *args):
        self.q.put(indata[::self.audio_cf.downsample, self.mapping])

    @property
    def sensitivity(self):
        return self.sensitivity_cf.level

    @sensitivity.setter
    def sensitivity(self, value):
        print('sensitivity is ' + str(value))
        self.sensitivity_cf.level = value

    def press(self, key, **kwargs):

        if key == 'r':
            self.order *= 0
            self.order += torch.randperm(self.L).to(DEVICE)

        if key == 't':
            self.fDrift -= 0.01
            if self.fDrift < 0:
                self.fDrift *= 0
            self.drift = (self.noise[:, self.L:self.LE] * .0).uniform_(-self.fDrift, self.fDrift)
            self.noise[:, self.L:self.LE] = self.noise[:, self.L:self.LE].uniform_(-1, 1)

        if key == 'z':
            self.fDrift += 0.01
            if self.fDrift > 0.2:
                self.fDrift *= 0
                self.fDrift += 0.2
            self.drift = (self.noise[:, self.L:self.LE] * .0).uniform_(-self.fDrift, self.fDrift)
            self.noise[:, self.L:self.LE] = self.noise[:, self.L:self.LE].uniform_(-1, 1)

        if key == 'i':
            self.audioA.nperseg += 100
            self.audioA.nfft = self.audioA.nperseg
            print("audio snapshots", self.audioA.getTFFT(
                int(self.audio_cf.window * 44100 / (1000 * self.audio_cf.downsample)),
            ))

        if key == 'u':
            self.audioA.nperseg -= 100
            self.audioA.nperseg = max(self.audioA.nperseg, 500)
            self.audioA.nfft = self.audioA.nperseg
            print("audio snapshots", self.audioA.getTFFT(
                int(self.audio_cf.window * 44100 / (1000 * self.audio_cf.downsample))),
            )

    def _press(self, event):
        self.press(event.key)

    def get_noise(self):
        if self.random_walk:
            alpha = (self.n_steps % self.walk_steps) / self.walk_steps
            noise = alpha * self.new_noise + (1 - alpha) * self.noise
            self.n_steps += 1
            if (self.n_steps % self.walk_steps < 1):
                self.noise = self.new_noise
                self.new_noise = self._init_noise()
            return noise
        else:
            return self.noise

    def create_sample(self, x):
        magnitude = self.audioA.sp3(x)
        magnitude = psp3(magnitude)
        magnitude = self.audioA.shrinkMel(magnitude)
        noise = self.get_noise()
        image = self.infer_step(magnitude, noise)
        return image

    def update(self):
        while True:
            try:
                data = self.q.get_nowait()
            except queue.Empty:
                break
            shift = len(data)
            self.plotdata = np.roll(self.plotdata, -shift, axis=0)
            self.plotdata[-shift:, :] = data
        return self.create_sample(self.plotdata)

    def __iter__(self):
        with self.stream:
            while True:
                yield self.update()
