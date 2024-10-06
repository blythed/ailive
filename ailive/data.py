import numpy as np
import os
import pickle
import torch

from PIL import Image
from torchvision import transforms


class LiveDataset:
    def __init__(self, path, batch_size=8):

        self.data = pickle.load(open(path, 'rb'))
        self.T = len(self.data)

        self.batch_size = batch_size

    def next(self, batchsize=None):
        if batchsize is None:
            batchsize = self.batch_size
        buf = []
        for z in range(batchsize):
            r = np.random.randint(self.T)
            v = self.data[r]
            if len(v.shape) == 2:
                r2 = np.random.randint(v.shape[1])
                v = v[:, r2]  # to give back a random frame
            if v.shape[0] == 60:
                v /= v.sum() + 1e-8 # ????
            buf += [v]
        buf = np.array(buf)
        return torch.FloatTensor(buf)


class TextureDataset:
    """Dataset wrapping images from a random folder with textures

    Arguments:
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, img_path, transform=None, resize=False):
        self.img_path = img_path
        self.transform = transform
        names = os.listdir(img_path)
        self.X_train = []
        for n in names[:]:
           if n[-3:] != 'jpg' and  n[-3:] != 'png':
               continue
           if n[-8:] == 'INDS.png':
               continue
           name = self.img_path + n
           img = Image.open(name)
           img = img.convert('RGB')  # fixes truncation?

           if resize:  # for flower
               img = transforms.Resize(size=160, interpolation=2)(img)

           self.X_train += [img]
           print (n, "img added", img.size, "total length", len(self.X_train))

        # TODO avoid hack, also change dataloader iterator length of each epoch
        # make custom, don't always count files
        # this affects epoch length..
        if len(self.X_train) < 2000:
            c = int(2000 / len(self.X_train))
            self.X_train *= c
        # print(self.X_train)

    def __getitem__(self, index):
        if False:
            name = self.img_path + self.X_train[index]
            img = Image.open(name)
        else:
            img = self.X_train[index]
        if self.transform is not None:
            img2 = self.transform(img)
        label = 0
        return img2, label

    def __len__(self):
        return len(self.X_train)
