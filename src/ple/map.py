import random
from functools import lru_cache

import numpy as np

import keras.backend as K
import skimage
from keras.models import load_model
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d


def masked_mse(target, pred):
    mask = K.cast(K.not_equal(target, -1), K.floatx())
    mse = K.mean(K.square((pred - target) * mask))
    return mse


def _pose(x, y, theta):
    return mktr(x, y) @ mkrot(theta)


def move_pose(x, y, theta, v, omega, dt=1):
    return getxytheta(_pose(x, y, theta) @ arc(v, omega, dt=dt))


def arc(v_lin, v_ang, dt=1):
    if(np.isclose(v_ang, 0)):  # we are moving straight, R is at the infinity and we handle this case separately
        return mktr(v_lin * dt, 0)  # note we translate along x
    R = v_lin / v_ang
    return mktr(0, R) @ mkrot(v_ang * dt) @ mktr(0, -R)


def read_image(path, scale_factor=1):
    "Reads an image taking into account the scalling and the bitdepth"
    hm = skimage.io.imread(path)
    if hm.ndim > 2:
        hm = skimage.color.rgb2gray(hm)
    elif hm.ndim == 2:
        if hm.dtype == 'uint8':
            divided = 255.0
        if hm.dtype == 'uint16':
            divided = 65535.0
        hm = hm / divided
    rescale = interp1d([0, np.max(hm)], [0, scale_factor])
    hm = rescale(hm)
    return hm


def extract_patch(img, x, y, theta, size):
    tf1 = skimage.transform.SimilarityTransform(translation=[-x, -y])
    tf2 = skimage.transform.SimilarityTransform(rotation=-theta)
    tf3 = skimage.transform.SimilarityTransform(scale=1.0)  # todo
    tf4 = skimage.transform.SimilarityTransform(translation=[size / 2, size / 2])
    trf = (tf1 + (tf2 + (tf3 + tf4))).inverse
    return skimage.transform.warp(img, trf, output_shape=(size, size), mode="edge")


def extract_patch_anymal(img, x, y, theta, size, output_size=None):
    patch = extract_patch(img, x, y, theta, size)
    # normalization based on the center of the patch;
    # and other transformations, e.g. resizing
    patch = patch - patch[patch.shape[0] // 2, patch.shape[1] // 2]
    if output_size is not None and size != output_size:
        patch = skimage.transform.resize(
            patch, (output_size, output_size), mode='constant', anti_aliasing=True)
    return patch


def extract_patch_thymio(img, x, y, theta, size):
    patch = extract_patch(img, x, y, theta, size)
    # rotate to place +x up, and +y right (in our robot frame)
    patch = skimage.transform.rotate(patch, 90, resize=True)
    return patch


def to_image(point, image_size, map_size):
    # we assume same resolution for x and y
    x, y = point
    res = image_size / map_size
    # we also assume hm's origin is at the top left corner without (-) part
    return ((x * res), (y * res))


def mktr(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def mkrot(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta),  np.cos(theta), 0],
                     [0,              0,             1]])


def getxytheta(f):
    return [
        f[0, 2],
        f[1, 2],
        np.arctan2(f[1, 0], f[0, 0])]


def point_in_frame(frame, point):
    return getxytheta(np.linalg.inv(_pose(*frame)) @ _pose(point[0], point[1], 0))[:2]


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


class Map:

    # The map coordinate frame is the same as an image, i.e. centered in the top-left corner
    # with x axis to the right and y axis to the bottom.

    def __init__(self, model_path, image, size=10, patch_size=100, **kwargs):
        self.image = image
        self.model = load_model(model_path, **kwargs)
        self.patch_size_px = patch_size
        self.image_size = self.image.shape[0]
        if size is None:
            self.size = self.image_size
        else:
            self.size = size

    def z(self, x, y):
        ix, iy = to_image((x, y), self.image_size, self.size)
        i = int(iy)
        j = int(ix)
        try:
            return float(self.image[i, j])
        except IndexError:
            return -1

    def plot(self, cmap=plt.cm.Greys, vmin=None, vmax=None):
        plt.imshow(self.image, extent=(0, self.size, self.size, 0),
                   cmap=cmap, vmin=vmin, vmax=vmax)

    def to_ros(self, x, y, theta=None, with_z=False):
        c = self.size / 2
        s = [x - c, c - y]
        if with_z:
            s += [self.z(x, y)]
        if theta is not None:
            s += [-theta]
        return s

    def from_ros(self, x, y, theta):
        c = self.size / 2
        return [x + c, c - y, -theta]


class ThymioMap(Map):

    def __init__(self, model_path, image_path, size=5, patch_size=80, **kwargs):
        image = np.load(image_path)
        # the original map has x right and y down
        image = image.reshape(image.shape[:2])[::-1, ::-1]
        super(ThymioMap, self).__init__(model_path=model_path, image=image, size=size,
                                        patch_size=patch_size, **kwargs)

    @lru_cache(maxsize=None)
    def traversable(self, pose, control, plot=False):

        p = to_image(pose[:2], self.image_size, self.size)
        patch = extract_patch_thymio(self.image, p[0], p[1], pose[2], self.patch_size_px)
        if plot:
            w = self.patch_size_px * self.size / self.image_size / 2
            plt.imshow(patch, extent=(pose[0] - w, pose[0] + w, pose[1] - w, pose[1] + w))
        patch = np.expand_dims(patch, axis=2)
        control = (control[0], -control[1])
        y_estimates = self.model.predict([np.array([patch]), np.array([control])])
        return (y_estimates[0][:, 0][0], y_estimates[1][0][0])

    def plot_traversability_points(self, pose, n=10, margin=0.25, moves=1, marker='.',
                                   threshold=0, arrows=False):
        x, y, theta = pose
        self.plot()
        plt.quiver([x], [y], [np.cos(-theta)], [np.sin(-theta)])
        for v in [0.08, -0.08]:
            for omega in np.linspace(-0.5, 0.5, n):
                next_pose = pose
                p = 1
                for _ in range(moves):
                    q, _ = self.traversable(tuple(next_pose), (v, omega), plot=False)
                    p *= q
                    if p < threshold:
                        break
                    next_pose = move_pose(next_pose[0], next_pose[1], next_pose[2], v, omega)
                    plt.plot(*next_pose[:2], marker, color=(1 - p, p, 0))
        plt.xlim(x - margin, x + margin)
        plt.ylim(y + margin, y - margin)


n = Normalize(0, 1, clip=True)


class AnymalMap(Map):

    def __init__(self, model_path, image_path, size=None, patch_size=100, z_scale=1.0, min_r=0.15,
                 max_r=0.5, **kwargs):
        image = read_image(image_path, z_scale)
        super(AnymalMap, self).__init__(model_path=model_path, image=image, size=size,
                                        patch_size=patch_size, **kwargs)
        self.min_r = min_r
        self.max_r = max_r

    @lru_cache(maxsize=None)
    def traversable(self, pose, target, frame='rel', plot=False):
        # transform target in pose frame

        if frame == 'abs':
            target = point_in_frame(pose, target)

        r = np.linalg.norm(target)
        if r < self.min_r or r > self.max_r:
            return (0, np.inf)

        p = to_image(pose[:2], self.image_size, self.size)
        patch = extract_patch_anymal(self.image, p[0], p[1], pose[2], self.patch_size_px)
        if plot:
            w = self.patch_size_px * self.size / self.image_size / 2
            plt.imshow(patch, extent=(pose[0] - w, pose[0] + w, pose[1] - w, pose[1] + w))
        patch = np.expand_dims(patch, axis=2)
        target = [target[0], -target[1]]
        y_estimates = self.model.predict([np.array([patch]), np.array([target])])
        return (y_estimates[0][:, 1][0], y_estimates[1][0][0])

    def t_color(self, pose, target, alpha=0.5):
        p, t = self.traversable(pose, target, frame='abs')
        if p == 0:
            return (0, 0, 0, 0)
        c = list(plt.cm.RdYlGn(n(p)))
        c[-1] = alpha
        return c

    def imt(self, pose, n=10, alpha=0.5):
        x0, y0, _ = pose
        m = self.max_r - 1e-3
        return [[self.t_color(pose, (x + x0, y + y0), alpha=alpha) for x in np.linspace(-m, m, n)]
                for y in np.linspace(-m, m, n)]

    def plot_traversability_points(self, pose, n=50, margin=1):
        xs = np.linspace(-self.max_r + 1e-3, self.max_r - 1e-3, n)
        ys = np.linspace(-self.max_r + 1e-3, self.max_r - 1e-3, n)
        self.plot()
        cp = np.array(pose[:2])
        for x in xs:
            for y in ys:
                target = (x, y)
                d = np.linalg.norm(np.array(target))
                if self.min_r <= d <= self.max_r:
                    target = cp + target
                    p, t = self.traversable(pose, tuple(target.tolist()), frame='abs')
                    plt.plot(*target,  '.', color=(1 - p, p, 0))
        plt.xlim(pose[0] - margin, pose[0] + margin)
        plt.ylim(pose[1] + margin, pose[1] - margin)

    def plot_traversability(self, pose, n=10, alpha=0.5, margin=1):
        self.plot()
        r = plt.imshow(self.imt(pose, n=n, alpha=alpha),
                       extent=(pose[0] - 0.5, pose[0] + 0.5, pose[1] + 0.5, pose[1] - 0.5))
        if margin:
            plt.xlim(pose[0] - margin, pose[0] + margin)
            plt.ylim(pose[1] + margin, pose[1] - margin)
        return r
