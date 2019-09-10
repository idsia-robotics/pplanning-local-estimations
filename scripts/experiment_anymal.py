import os
import random
import sys
from datetime import datetime as dt
from itertools import repeat
from multiprocessing import Pool

import numpy as np

import skimage
import yaml
from ompl import geometric as og
from ple.anymal_planner import Plan
from ple.map import AnymalMap, masked_mse, to_image


def pixel(img, x, y, image_size, size):
    ix, iy = to_image((x, y), image_size, size)
    i = int(iy)
    j = int(ix)
    return img[i, j]


def random_point(can_sample_img, m):
    while True:
        x = random.random() * m.size
        y = random.random() * m.size
        if pixel(can_sample_img, x, y, m.image_size, m.size):
            return (x, y)


def random_pair(can_sample_img, m, min_dist=3):
    sx, sy = random_point(can_sample_img, m)
    while True:
        tx, ty = random_point(can_sample_img, m)

        if np.linalg.norm(np.array((sx, sy)) - np.array((tx, ty))) >= min_dist:
            break
    st = random.random() * 2 * np.pi
    return (sx, sy, st), (tx, ty)


def f(s):
    index, folder, data = s
    os.mkdir(f'{folder}/{index}')

    if data['masked_mse']:
        custom_objects = {'masked_mse': masked_mse}
    else:
        custom_objects = {}

    m = AnymalMap(image_path=data['image_path'],
                  model_path=data['model_path'],
                  size=data['size'],
                  z_scale=data['z_scale'],
                  custom_objects=custom_objects)
    sample_img = skimage.io.imread(data['sample_image_path'])
    can_sample_img = (sample_img[..., 1] != sample_img[..., 0])
    s, t = random_pair(can_sample_img, m, min_dist=data['min_dist'])
    p = Plan(m, s, t, max_time=data['duration'],
             objective=data['objective'], threshold=data['threshold'], planner=og.RRTstar)
    p.solve(data['duration'], max_time=data['duration'])
    p.save(f'{folder}/{index}')
    ros = p.map.to_ros
    source = ros(*p.s, with_z=True)
    return index, source


def main(config):

    name = config.split('.')[0]
    with open(config) as g:
        data = yaml.load(g)

    folder = os.path.join(os.path.dirname(__file__), 'results', f'{name}_{dt.now()}')
    os.mkdir(folder)

    with open(f'{folder}/experiment.yaml', 'w') as e:
        e.write(yaml.dump(data))

    samples = data['samples']
    initial_sample = data['initial_sample']

    with Pool(data['number_of_processes']) as p:
        rs = p.map(f, zip(range(initial_sample, initial_sample + samples),
                          repeat(folder), repeat(data)))
    data = '\n'.join([f'{i} {x:.3f} {y:.3f} {z:.3f} {t:.3f}' for i, (x, y, z, t) in rs])
    with open(f'{folder}/sources.txt', 'w') as g:
        g.write(data)


if __name__ == '__main__':
    main(sys.argv[1])
