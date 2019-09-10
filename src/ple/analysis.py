import itertools
import os
from multiprocessing import Pool

import numpy as np

import keras.backend as K
import pandas as pd
import yaml
from ple.map import AnymalMap
from tqdm import tqdm_notebook as tqdm


def load_file(args):
    i, f = args
    with open(f) as g:
        data = yaml.load(g)
        x, y, theta = zip(*data['path']['poses'])
        return pd.DataFrame(
            {'segment_id': list(range(1, 1 + len(data['path']['durations']))),
             'pred_time': data['path']['durations'],
             'x0': x[:-1],
             'y0': y[:-1],
             'theta0': theta[:-1],
             'x1': x[1:],
             'y1': y[1:],
             'theta1': theta[1:],
             'path_id': int(i),
             'p': data['path']['probabilities']
             })


def load_plans(folder, map_=None, pool=7, use_real_pose=False, model=None):
    files = [(i, f) for (i, f) in [(i, os.path.join(folder, i, 'solution.yaml'))
                                   for i in os.listdir(folder)]
             if os.path.exists(f)]
    with Pool(pool) as p:
        df = pd.concat(tqdm(p.imap_unordered(load_file, files), total=len(files)))
    if map_ is not None and model is not None:
        c = map_.from_ros

        def pred(r):
            s = tuple(c(r.x0, r.y0, r.theta0))
            t = tuple(c(r.x1, r.y1, r.theta1)[:2])
            r['p'], r['pred_time'] = map_.traversable(s, t, frame='abs')
            return r
        df = df.apply(pred, axis=1)
    return df


def load_experiment(plan_folder, realization_path, map_=None, use_real_pose=False, model=None,
                    diff=-1):
    data = load_plans(plan_folder, map_, use_real_pose=use_real_pose, model=model)
    e_data = pd.read_csv(realization_path)

    e_data.rename(columns={'pos_sim_z': 'z1_r', 'pos_sim_x': 'x1_r', 'pos_sim_y': 'y1_r', 'pos_sim_yaw': 'theta1_r'},
                  inplace=True)
    e_data['x0_r'] = e_data.x1_r.shift(1, axis=0)
    e_data['y0_r'] = e_data.y1_r.shift(1, axis=0)
    e_data['z0_r'] = e_data.z1_r.shift(1, axis=0)
    e_data['theta0_r'] = e_data.theta1_r.shift(-1, axis=0)
    if 'd_time' not in e_data:
        e_data['d_time'] = e_data['time'].diff()
    e_data['segment_id'] = e_data['pose_id'] + diff
    e_data['success'][(e_data['success'].diff() == 0.0) & (e_data['success'] == 0.0)] = 2.0
    cs_r = ['path_id', 'segment_id']
    cs_l = ['path_id', 'segment_id']
    df = pd.merge(data, e_data, left_on=cs_l, right_on=cs_r)
    if map_ is not None and use_real_pose:
        c = map_.from_ros

        def pred(r):
            t = tuple(c(r.x1, r.y1, r.theta1)[:2])
            tr = tuple(c(r.x1_r, r.y1_r, r.theta1_r)[:2])
            sr = tuple(c(r.x0_r, r.y0_r, r.theta0_r))
            pr, dr = map_.traversable(sr, t, frame='abs')
            if pr > 0 and np.isfinite(dr):
                r['p_r'], r['pred_time_r'] = pr, dr
            else:
                r['p_r'], r['pred_time_r'] = np.NaN, np.NaN
            prr, drr = map_.traversable(sr, tr, frame='abs')
            if prr > 0 and np.isfinite(drr):
                r['p_rr'], r['pred_time_rr'] = prr, drr
            else:
                r['p_rr'], r['pred_time_rr'] = np.NaN, np.NaN
            return r
        df = df.apply(pred, axis=1)

    return df


def subpath(df, path_id, i, j, last, use_real_pose=False):
    if i == 0 or df.iloc[i - 1].success == 1.0:
        # success = df.iloc[j-1]['success']
        complete = (i == 0 and j == last)
        data = {'complete': complete,
                'path_id': path_id,
                'start_pose': i,
                'end_pose': j,
                'pred_time': df[i: j]['pred_time'].sum(),
                'time': df[i: j]['d_time'].sum(),
                'p': df[i: j]['p'].product(),
                'success': (df[i: j]['success'] == 1.0).product()}
        if use_real_pose:
            data['pred_time_r'] = df[i + 1: j]['pred_time'].sum() + df[i:i + 1]['pred_time_r'].sum()
            data['p_r'] = df[i + 1: j]['p'].product() * df[i:i + 1]['p_r'].product()
            data['pred_time_rr'] = df[i: j]['pred_time_rr'].sum()
            data['p_rr'] = df[i: j]['p_rr'].product()
        return pd.DataFrame(data, index=[0])
    else:
        return pd.DataFrame()


def path(data, path_id, use_real_pose=False):
    df = data[data.path_id == path_id]
    return subpath(df, path_id, 0, len(df), len(df), use_real_pose=use_real_pose)


def all_subpaths(data, path_id, use_real_pose=False):
    df = data[data.path_id == path_id]
    last = len(df)
    return pd.concat(subpath(df, path_id, i, j, last, use_real_pose=use_real_pose)
                     for i, j in itertools.combinations(range(last + 1), 2))


def compute_all_subpaths(data, pool=7, use_real_pose=False):
    ids = data['path_id'].unique()
    args = zip(itertools.repeat(data), ids, itertools.repeat(use_real_pose))
    with Pool(pool) as p:
        df = pd.concat(tqdm(p.imap_unordered(f, args), total=len(ids)))
    return df


def f(arg):
    data, i, use_real_pose = arg
    return all_subpaths(data, i, use_real_pose)


def masked_mse(target, pred):
    mask = K.cast(K.not_equal(target, -1), K.floatx())
    mse = K.mean(K.square((pred - target) * mask))
    return mse


def stats_success(df, n=20, name='path_test'):
    df['p_bin'] = pd.cut(df['p'], n)
    data = df.groupby('p_bin')[['success', 'p']]
    mean = data.mean()
    mean.to_csv(f'{name}_subpaths_success_mean.csv')


def stats_time(df, n=3.0, name='path_test'):
    df = df[df.success == 1.0]
    df['ti'] = round(df.pred_time / n)
    data = df.groupby('ti')[['pred_time', 'time']]
    mean = data.mean()
    std = data.std()
    d = pd.DataFrame({'pred_time': mean.pred_time, 'time_p': mean.time +
                      std.time, 'time_m': mean.time - std.time})
    d.to_csv(f'{name}_subpaths_time.csv')


def analyse_experiment(folder, results, model=None, model_name='',
                       pool=7, use_real_pose=False, diff=0, save_to_folder='.'):
    with open(os.path.join(folder, 'experiment.yaml')) as f:
        exp_data = yaml.load(f)
    map_name = exp_data['image_path'].split('/')[-1].split('.')[0]
    segments_path = os.path.join(save_to_folder, f'{map_name}_segments_{model_name}.csv')
    print(save_to_folder, map_name, model_name, segments_path)
    if not os.path.exists(segments_path):
        if use_real_pose and not model:
            m = AnymalMap(exp_data['model_path'],
                          os.path.join('maps', exp_data['image_path']),
                          z_scale=exp_data['z_scale'],
                          size=exp_data['size'],
                          custom_objects={'masked_mse': masked_mse})
        elif model is not None:
            m = AnymalMap(model,
                          os.path.join('maps', exp_data['image_path']),
                          z_scale=exp_data['z_scale'],
                          size=exp_data['size'],
                          custom_objects={'masked_mse': masked_mse})
        else:
            m = None

        data = load_experiment(folder, results, map_=m, model=model, use_real_pose=use_real_pose,
                               diff=diff)
        data.to_csv(segments_path)
    else:
        data = pd.read_csv(segments_path)
    subpaths_path = os.path.join(save_to_folder, f'{map_name}_subpaths_{model_name}.csv')
    if not os.path.exists(subpaths_path):
        data_subpaths = compute_all_subpaths(data, pool=pool, use_real_pose=use_real_pose)
        data_subpaths.to_csv(subpaths_path)
    else:
        data_subpaths = pd.read_csv(subpaths_path)
    return data, data_subpaths
