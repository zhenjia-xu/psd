import os

import numpy as np
from torch.utils.data import Dataset
from utils import imread

from utils import imresize


class MotionDataset(Dataset):
    def __init__(self, data_path, split, size, scale):
        self.data_path = data_path
        self.split = split
        self.size = size
        self.scale = scale
        self.data = open(os.path.join(self.data_path, '{0}.txt'.format(self.split))).read().splitlines()[:10000]

    def __getitem__(self, index):
        images = []
        for k in range(2):
            image = imread(os.path.join(self.data_path, '{0}_im{1}.png'.format(
                self.data[index], k + 1)))
            image = imresize(image, size=self.size)
            images.append(image)

        tmp = np.load(os.path.join(self.data_path, '{0}_f.npy'.format(self.data[index]))).astype(np.float32)

        flow_inputs = []
        for axis in range(2):
            input = imresize(tmp[axis], size=self.size)
            flow_inputs.append(input)

        flow_targets = []
        for axis in range(2):
            target = imresize(tmp[axis], size=self.size) * self.scale
            flow_targets.append(target)

        image_inputs = images[0]
        flow_inputs = np.stack(flow_inputs, axis=0)

        image_targets = images[1]
        flow_targets = np.stack(flow_targets, axis=0)

        returns = {
            'image_inputs': image_inputs.astype(np.float32),
            'flow_inputs': flow_inputs.astype(np.float32),
            'image_targets': image_targets.astype(np.float32),
            'flow_targets': flow_targets.astype(np.float32),
        }
        return returns

    def __len__(self):
        return len(self.data)