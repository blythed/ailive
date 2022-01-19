import numpy
import os
import PIL.Image
import pickle
import torch


class AudioDataset:
    def __init__(self, path: str, batch_size: int = 8, use_projection: bool = True):
        self.data = pickle.load(open(path, 'rb'))
        self.n_time_points = len(self.data)
        self.batch_size = batch_size
        self.use_projection = use_projection

    def next(self):
        buf = []
        for z in range(self.batch_size):
            r = numpy.random.randint(self.n_time_points)
            v = self.data[r]
            if len(v.shape) == 2:
                r2 = numpy.random.randint(v.shape[1])
                v = v[:, r2]  # to give back a random frame
            if not self.use_projection:
                v /= v.sum() + 1e-8
            buf += [v]
        buf = numpy.array(buf)
        return torch.FloatTensor(buf)


class TextureDataset:
    def __init__(self, img_path: str, transform: callable):
        if not img_path.endswith('/'):
            img_path += '/'
        self.img_path = img_path
        self.transform = transform
        names = os.listdir(img_path)
        self.x_train = []
        for name in names[:]:
            if not name.endswith('.jpg') and not name.endswith('png'):
               continue
            name = self.img_path + name
            img = PIL.Image.open(name)
            img = img.convert('RGB')
            self.x_train += [img]
        print(name, "img added", img.size, "total length", len(self.x_train))
        if len(self.x_train) < 2000:
            c = int(2000 / len(self.x_train))
            self.x_train *= c

    def __getitem__(self, index):
        img = self.x_train[index]
        img = self.transform(img)
        return img, 0

    def __len__(self):
        return len(self.x_train)
