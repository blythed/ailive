import argparse
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import sys
import time
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from ailive import DEVICE
from ailive.data import TextureDataset, LiveDataset
from ailive.models import Generator, Discriminator, setWave, setNoise
from ailive.config import Struct
from ailive.generate import generate
from ailive.config import cf

from shu import shu
from shu.core.db import db


matplotlib.use('Agg')
cudnn.benchmark = True


class Logger:
    """
    Class to log weights during training.
    :param model: model whose weights should be logged
    :param save_: path to checkpoint directory
    """
    def __init__(self,
                 model,
                 save_):

        self.model = model
        self.save_ = save_
        self.variables = [x for x in self.model.state_dict().keys()]

    def init(self):
        """
        Initialize logger by setting up weight dictionaries
        """
        self.save_dict = dict(zip(self.variables,
                                  [[] for _ in self.variables]))
        self.index_dict = {}

        for var_ in self.variables:
            tensor_ = self.model.state_dict()[var_].view(-1)
            len_ = len(tensor_)
            ix = [random.choice(range(len_)) for _ in range(5)]
            self.index_dict[var_] = ix

    def dump(self):
        """
        Print logger to .json file with path in self.save_
        """
        with open(self.save_ + "/weights.json", "w") as f:
            json.dump(self.save_dict, f)

    def log(self):
        """
        Log weights into weight dictionaries at some point in training.
        """
        for var_ in self.variables:
            tensor_ = self.model.state_dict()[var_].view(-1)
            to_append = [float(tensor_[choice])
                         for choice in self.index_dict[var_]]
            self.save_dict[var_].append(to_append)
            if len(self.save_dict[var_]) > 1000:
                self.save_dict[var_] = \
                    self.save_dict[var_][-1000:]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_configs(path, dataroot, name):
    with open(path) as f:
        cf = yaml.load(f, Loader=yaml.FullLoader)

    cf['train']['dataroot'] = dataroot
    return Struct(**cf['train']), \
           Struct(**cf['generator']), \
           Struct(**cf['discriminator'])


class Trainer:
    def __init__(self, shu_name, train_cf, gen_cf, dis_cf):

        if shu_name[-1] != '/':
            shu_name += '/'
        self.shu_name = shu_name
        self.train_cf = train_cf
        self.gen_cf = gen_cf
        self.dis_cf = dis_cf
        self.epoch = 0
        self.it = 0
        self.glance_dir = 'glances/' + self.shu_name

        os.system(f'mkdir -p {self.glance_dir}')

        self._init_seed()
        self.transform = self._init_transforms()
        if train_cf.fRec > 0:
            self.recon = True
            self.netD, self.netG, self.netR = self._init_model()
        else:
            self.recon = False
            self.netD, self.netG = self._init_model()
        self._init_weights()
        if self.train_cf._get('load', False):
            self._load_weights()
        self.optimizerD, self.optimizerG = self._get_optimizers()
        self.dataloader, self.dataset, self.cdataset = self._get_data()
        self.criterion = nn.BCELoss()
        self.logger = Logger(self.netG, 'checkpoints/' + self.shu_name)
        self.logger.init()

        nz_other = train_cf.imageSize // 2 ** gen_cf.nDep
        self.noise = torch.FloatTensor(train_cf.batchSize,
                                       gen_cf.nz,
                                       nz_other,
                                       nz_other).to(DEVICE)
        self.noise = setNoise(self.noise, nGL=self.gen_cf.nGL)

        self.label = torch.FloatTensor(train_cf.batchSize).to(DEVICE)
        self.REAL_LABEL = 1
        self.FAKE_LABEL = 0

    def _init_seed(self):
        random.seed(self.train_cf._get('manualSeed', int(time.time())))
        torch.manual_seed(self.train_cf._get('manualSeed', int(time.time())))

    def _init_transforms(self):
        return transforms.Compose([
            transforms.RandomCrop(self.train_cf.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _init_model(self):
        netD = Discriminator(self.dis_cf.ndf, self.gen_cf.nDep)
        netG = Generator(
            self.gen_cf.ngf,
            self.gen_cf.nDep,
            self.gen_cf.nz,
            self.gen_cf.nGL,
            Ctype=self.gen_cf.Ctype,
            nPeriodic=self.gen_cf.nperiodic,
            nc=self.gen_cf.nc,
            Kperiodic=self.gen_cf.Kperiodic
        )
        netD.to(DEVICE)
        netG.to(DEVICE)
        if self.recon:
            netR = Reconstructor(gen_cf.ndf, gen_cf.nDep, gen_cf.nGL)
            netR.to(DEVICE)
            return netD, netG, netR
        else:
            return netD, netG

    def _init_weights(self):
        self.netD.apply(weights_init)
        self.netG.apply(weights_init)
        if self.recon:
            self.netR.apply(weights_init)

    def _load_weights(self):
        self.netG.load_state_dict(torch.load(train_cf.load))

    def _get_optimizers(self):
        optimizerD = optim.Adam(
            self.netD.parameters(),
            lr=self.train_cf.lr,
            betas=(self.train_cf.beta1, 0.999)
        )

        if self.recon:
            optimizerG = optim.Adam(
                list(self.netG.parameters()) + list(self.netR.parameters()),
                lr=self.train_cf.lr,
                betas=(self.train_cf.beta1, 0.999)
            )
        else:
            optimizerG = optim.Adam(
                self.netG.parameters(),
                lr=self.train_cf.lr,
                betas=(self.train_cf.beta1, 0.999)
            )
        return optimizerD, optimizerG

    def _get_data(self):
        dataset = TextureDataset(self.train_cf.dataroot, self.transform)
        cdataset = LiveDataset(self.train_cf.audio_path,
                               self.train_cf.batchSize)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.train_cf.batchSize,
            shuffle=True,
            num_workers=1
        )
        return dataloader, dataset, cdataset

    def _take_discriminator_step(self):
        self.netD.zero_grad()
        output = self.netD(self.text)
        errD_real = \
            self.criterion(output, output.detach() * 0 + self.REAL_LABEL)
        errD_real.backward()
        self.D_x = output.mean().item()
        with torch.no_grad():
            fake = self.netG(self.noise)
        output = self.netD(fake.detach())
        errD_fake = \
            self.criterion(output, output.detach() * 0 + self.FAKE_LABEL)
        errD_fake.backward()
        self.D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        self.optimizerD.step()

    def _take_generator_step(self):
        self.optimizerG.zero_grad()
        self.content = self.cdataset.next().to(DEVICE)
        self.noise = setNoise(self.noise, audio=self.content, nGL=self.gen_cf.nGL)
        self.fake = self.netG(self.noise)
        output = self.netD(self.fake)
        if self.recon:
            recZ = self.netR(self.fake)
            errR = \
                (((recZ - self.noise[:, :self.gen_cf.nGL, :1, :1])) ** 2).mean()
        else:
            errR = self.noise.sum() * 0
        errG = self.criterion(output, output.detach() * 0 + self.REAL_LABEL) + \
               self.train_cf.fRec * errR
        errG.backward()
        D_G_z2 = output.mean().item()
        self.optimizerG.step()

    def _make_glances(self):

        vutils.save_image(
            self.text * 0.5 + 0.5,
            f'{self.glance_dir}/real_textures.jpg',
            normalize=False
        )
        vutils.save_image(
            self.content * 0.5 + 0.5,
            f'{self.glance_dir}/real_contents.jpg',
            normalize=False
        )
        vutils.save_image(
            self.fake * 0.5 + 0.5,
            f'{self.glance_dir}/tex_{self.epoch:03}.jpg',
            normalize=False
        )

        n2 = self.noise[:4].repeat(1, 1, 3, 3)
        n2 = setNoise(n2, audio=self.content[:4], nGL=self.gen_cf.nGL)
        self.netG.eval()
        with torch.no_grad():
            fake2 = self.netG(n2)
        self.netG.train()
        vutils.save_image(
            fake2 * 0.5 + 0.5,
            f'{self.glance_dir}/tex2_{self.epoch:03}.jpg',
            normalize=False,
        )

        glance = {"tex": f'/static/{self.glance_dir}tex_{self.epoch:03}.jpg'}
        print(f'GLANCE 0 iteration: {self.it}; data: {json.dumps(glance)};')
        glance = {"tex2": f'/static/{self.glance_dir}tex2_{self.epoch:03}.jpg'}
        print(f'GLANCE 1 iteration: {self.it}; data: {json.dumps(glance)};')

        if self.gen_cf.nperiodic > 0:
            n2 = setWave(n2, self.gen_cf.nperiodic, self.netG)
            vutils.save_image(
                n2.view(-1, 1, n2.shape[2], n2.shape[3]) * .5 + 0.5,
                f'{self.glance_dir}/noise2_{self.epoch:03}.jpg',
                normalize=False,
            )

            glance = {"noise2":
                      f'/static/{self.glance_dir}noise2_{self.epoch:03}.jpg'}
            print(f'GLANCE 2 iteration: {self.it}; data: {json.dumps(glance)};')

            waves = n2[:, n2.shape[1] - self.gen_cf.nperiodic:].contiguous()
            waves = waves.view(-1, 1, n2.shape[2], n2.shape[3])
            vutils.save_image(
                waves * .5 + 0.5,
                f'{self.glance_dir}/waves_{self.epoch:03}.jpg',
                normalize=False,
            )
            glance = {"waves":
                      f'/static/{self.glance_dir}waves_{self.epoch:03}.jpg'}
            print(f'GLANCE 3 iteration: {self.it}; data: {json.dumps(glance)};')

    def _save_model(self):
        print('saving...')
        torch.save(self.netG.state_dict(),
                   f'checkpoints/{self.shu_name}/model.pt')
        self.logger.dump()

    def _do_iteration(self, batch):
        content = self.cdataset.next().to(DEVICE)
        self.text, _ = batch
        self.text = self.text.to(DEVICE)
        self.noise = setNoise(self.noise, audio=content, nGL=self.gen_cf.nGL)

        self._take_discriminator_step()
        self._take_generator_step()

        if self.it % 25 == 0:
            self.logger.log()

        if self.it % 5 == 0:
            print(f'TRAIN iteration: {self.it}; epoch: {self.epoch}; '
                  f'd_loss: {self.D_x}; g_loss: {self.D_G_z1};')

        self.it += 1

    def _do_epoch(self):
        for batch in self.dataloader:
            self._do_iteration(batch)
        if self.epoch % 5 == 0:
            self._make_glances()
        self._save_model()
        self.epoch += 1

    def _create_video(self):
        generate(self.gen_cf, cf.audio, self.shu_name)
        if self.gen_cf.nperiodic > 0:
            n = 4
        else:
            n = 2
        to_print = json.dumps({'example': '/static/' + self.glance_dir + 'example.mp4'})
        print(f'GLANCE {n} iteration: {self.it}; data: {to_print};')

    def _push_data(self):
        version = str(db.instances.find_one({'name': 'ailive'})['version'])
        print('pushing training data to ailive server')
        shu.files.sync(version)
        shu.files.checkpoints(version, exclude='"*"', include='"*/log"')
        shu.files.data(version, exclude='"data/audio/*"')

        ims = os.listdir('data/images/' + self.shu_name)
        ims = ['/static/images/' + x for x in ims]

        r = {'name': self.shu_name, 'examples': ims, 'info': {}, 'project': 'ailive'}
        db.datasets.replace_one({'name': self.shu_name}, r, upsert=True)


    def train(self):
        for _ in range(self.train_cf.n_epochs):
            self._do_epoch()
        self._create_video()
        self._push_data()
