import os
import sys
from datetime import datetime as dt
from itertools import product, repeat
from multiprocessing import Pool

import numpy as np

import yaml
from ple.map import ThymioMap
from ple.thymio_planner import Plan


def door(w, h=100):
    hw = w // 2
    hh = h // 2
    img = np.zeros((800, 800))
    img[(400 - hh):(400 + hh), :(400 - hw)] = 1
    img[(400 - hh):(400 + hh), (400 + hw):] = 1
    return img


def f(args):
    i, ((s, t), data) = args
    folder = data['folder']
    thymio_map = ThymioMap(image_path=data['img_path'],
                           model_path=data['model_path'],
                           size=data['size'])
    p = Plan(thymio_map, pose=s, target=t, tolerance=data['tolerance'],
             threshold=data['threshold'], allow_moving_backward=False, objective=data['objective'], k=data['k'])
    p.solve(data['duration'], data['duration'])
    os.mkdir(f'{folder}/{i}')
    p.save(f'{folder}/{i}', ros=False)


def main(config):

    name = config.split('.')[0]
    with open(config) as g:
        data = yaml.load(g)

    samples = data['samples']
    source = data['source']
    target = data['target']
    sources = product(source['x'], source['y'], source['theta'])
    targets = product(target['x'], target['y'], target['theta'])
    width = data['width']

    folder = os.path.join(os.path.dirname(__file__), 'results', f'{name}_{dt.now()}')
    os.mkdir(folder)

    with open(f'{folder}/experiment.yaml', 'w') as e:
        e.write(yaml.dump(data))

    if data['map_type'] == 'door':
        img = door(width)
        data['img_path'] = os.path.join(folder, 'map.npy')
        data['folder'] = folder
        np.save(data['img_path'], img)

    pairs = list(product(sources, targets)) * samples

    with Pool(data['number_of_processes']) as p:
        p.map(f, enumerate(zip(pairs, repeat(data))))


if __name__ == '__main__':
    main(sys.argv[1])
