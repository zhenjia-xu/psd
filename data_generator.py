import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
from numpy.random import randint
from tqdm import tqdm

from utils import flow2im, imresize, imwrite
from utils import mkdir


class Node:
    def __init__(self, offset, motion, size, shape, color, parent=None):
        self.offset = np.array(offset)
        self.motion = np.array(motion)
        self.size = size
        self.shape = shape.upper()
        self.color = color

        self.childs = []
        self.parent = parent
        if self.parent is not None:
            self.parent.childs.append(self)

    def get_offset(self):
        offset = self.offset.copy()
        if self.parent is not None:
            offset += self.parent.get_offset()
        return offset

    def get_motion(self):
        motion = self.motion.copy()
        if self.parent is not None:
            motion += self.parent.get_motion()
        return motion

    def step(self):
        self.offset += self.motion
        for child in self.childs:
            child.step()

    def render(self, image, flow):
        offset = self.get_offset()
        motion = self.get_motion()
        pixels = np.zeros_like(image)

        if self.shape == 'CIRCLE':
            cv2.circle(pixels, tuple(offset), self.size //
                       2, (1, 1, 1), thickness=-1)

        if self.shape == 'SQUARE':
            a = (offset[0] - self.size // 2, offset[1] - self.size // 2)
            b = (offset[0] + self.size // 2, offset[1] + self.size // 2)
            cv2.rectangle(pixels, a, b, (1, 1, 1), thickness=-1)

        if self.shape == 'TRIANGLE':
            a = (offset[0], offset[1] - self.size * np.sqrt(3) / 3)
            b = (offset[0] - self.size // 2,
                 offset[1] + self.size * np.sqrt(3) / 6)
            c = (offset[0] + self.size // 2,
                 offset[1] + self.size * np.sqrt(3) / 6)
            cv2.fillConvexPoly(pixels, np.array(
                [a, b, c], np.int32).reshape(-1, 1, 2), (1, 1, 1))

        mask = np.sum(pixels, 2) > 0
        for dim, color in enumerate(self.color):
            pixels[..., dim] *= color

        image *= np.stack([mask] * 3, -1) == 0
        for dim in range(3):
            image[..., dim] += mask * pixels[..., dim]

        flow *= np.stack([mask] * 2, -1) == 0
        for axis in range(2):
            flow[..., axis] += mask * motion[axis]

        for child in self.childs:
            child.render(image, flow)


class Canvas:
    def __init__(self, size, color):
        self.size = size
        self.color = color
        self.objs = []

    def add(self, obj):
        self.objs.append(obj)

    def step(self):
        for obj in self.objs:
            obj.step()

    def render(self, size=None):
        image = np.zeros((self.size, self.size, 3), np.uint8)
        flow = np.zeros((self.size, self.size, 2), np.float32)

        image[..., :] = self.color
        for obj in self.objs:
            obj.render(image, flow)

        if size is not None:
            image = imresize(image, size=size)
            flow = imresize(flow, size=size) / self.size * size

        return image, flow


def process(index):
    np.random.seed()

    size, c = 512, 75
    size_min, size_max = int(size * .2), int(size * .3)
    offset, motion = int(size * .2), int(size * .1)

    # colors
    colors = [
        (randint(c), randint(c), 255 - randint(c)),
        (randint(c), 255 - randint(c), randint(c)),
        (255 - randint(c), randint(c), randint(c))
    ]
    np.random.shuffle(colors)

    # canvas
    canvas = Canvas(
        size=size,
        color=(255 - randint(c), 255 - randint(c), 255 - randint(c))
    )

    # objects
    mode = randint(10)

    circle_motion = [randint(-motion, motion)] * 2

    circle = Node(
        offset=randint(-offset, offset, 2) + size // 2,
        motion=circle_motion,
        size=randint(size_min, size_max),
        shape='CIRCLE',
        color=colors[0]
    )
    canvas.add(circle)

    if not mode in [0, 1, 2]:
        square_motion = [0, randint(-motion, motion)]
        Node(
            offset=randint(-offset, offset, 2),
            motion=square_motion,
            size=randint(size_min, size_max),
            shape='SQUARE',
            color=colors[1],
            parent=circle
        )

    if not mode in [0, 3, 4]:
        triangle_motion = [randint(-motion, motion), 0]
        Node(
            offset=randint(-offset, offset, 2),
            motion=triangle_motion,
            size=randint(size_min, size_max),
            shape='TRIANGLE',
            color=colors[2],
            parent=circle
        )

    # render
    images, flows, masks = [], [], []
    meta_flows = []
    for step in range(2):
        image, flow = canvas.render(size=128)
        images.append(image / 255.)
        flows.append(flow / 128.)
        canvas.step()

    flow = flows[0].transpose(2, 0, 1)
    assert np.min(flow) >= -1 and np.max(flow) <= 1

    # save
    for step, image in enumerate(images):
        imwrite(os.path.join(data_path, '{0}_im{1}.png'.format(index, step + 1)), image)

    np.save(os.path.join(data_path, '{0}_f.npy'.format(index)), flow)


if __name__ == '__main__':
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/shape3-demo')
    parser.add_argument('--workers', default=32, type=int)

    # arguments
    args = parser.parse_args()
    print('==> arguments parsed')
    for key in vars(args):
        print('[{0}] = {1}'.format(key, getattr(args, key)))

    # data path
    data_path = args.data_path
    mkdir(data_path, clean=True)

    # tasks
    num_demo = int(1e3)

    tasks = []
    with open(os.path.join(data_path, 'demo.txt'), 'w') as fp:
        for k in range(num_demo):
            print(k + 1, file=fp)
            tasks.append(k + 1)

    # process
    pool = Pool(processes=args.workers)
    with tqdm(total=len(tasks)) as progress:
        for _ in pool.imap_unordered(process, tasks):
            progress.update()
