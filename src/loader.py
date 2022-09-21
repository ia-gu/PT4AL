import glob
from re import I
from PIL import Image

from torch.utils.data import Dataset
import torch
import numpy as np
import random

# dataloader for pretext tasks
class RotationLoader(Dataset):
    def __init__(self, x, y, path, transform=None):
        self.transform = transform
        self.x = x
        self.y = y
        self.path = path
        print("a",len(self.x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        path = self.path[idx]

        x = self.transform(x)
        x1 = torch.rot90(x, 1, [1,2])
        x2 = torch.rot90(x, 2, [1,2])
        x3 = torch.rot90(x, 3, [1,2])
        xs = [x, x1, x2, x3]
        rotations = [0,1,2,3]
        random.shuffle(rotations)
        return xs[rotations[0]], xs[rotations[1]], xs[rotations[2]], xs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], path.decode('utf-8')

class Loader2(Dataset):
    def __init__(self, x, y, path, is_train=True, transform=None, path_list=None):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        self.x = x
        self.y = y
        self.path = path

    def __len__(self):
        if(not self.is_train):
            return len(self.x)
        return len(self.path_list)

    def __getitem__(self, idx):
        if(not self.is_train):
            img = self.x[idx]
            label = self.y[idx]
            img = self.transform(img)

            return img, label
        
        # nidx = idx
        idx = self.path_list[idx].rstrip('\n').encode('utf-8')
        idx = np.where(idx == self.path)
        img = self.x[idx[0][0]]
        img = self.transform(img)
        label = self.y[idx[0][0]]

        return img, label