import numpy
import os
import PIL.Image
import pickle
import torch


class LiveDataset:
    def __init__(self, path, batch_size=8):
        self.data = pickle.load(open(path, 'rb'))
        self.n_time_points = len(self.data)
        self.batch_size = batch_size

    def next(self, batchsize=None):
        if batchsize is None:
            batchsize = self.batch_size
        buf = []
        for z in range(batchsize):
            r = numpy.random.randint(self.n_time_points)
            v = self.data[r]
            if len(v.shape) == 2:
                r2 = numpy.random.randint(v.shape[1])
                v = v[:, r2]  # to give back a random frame
            if v.shape[0] == 60:  # WTF???
                v /= v.sum() + 1e-8
            buf += [v]
        buf = numpy.array(buf)
        return torch.FloatTensor(buf)


class TextureDataset:
    """Dataset wrapping images from a random folder with textures

    Arguments:
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform
        names = os.listdir(img_path)
        self.X_train = []
        for n in names[:]:
           if not n[-3:].endswith('.jpg') and not n[-3:].endswith('png'):
               continue
           name = self.img_path + n
           img = PIL.Image.open(name)
           img = img.convert('RGB')
           self.X_train += [img]
           print (n, "img added", img.size, "total length", len(self.X_train))

    def __getitem__(self, index):
        img = self.X_train[index]
        img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.X_train)
