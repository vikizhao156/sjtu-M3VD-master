
import pandas as pd
import numpy as np



def resize(voxel, spacing, new_spacing=[1., 1., 1.]):
    '''Resize `voxel` from `spacing` to `new_spacing`.'''
    resize_factor = []
    for sp, nsp in zip(spacing, new_spacing):
        resize_factor.append(float(sp) / nsp)
    resized = scipy.ndimage.interpolation.zoom(voxel, resize_factor, mode='nearest')
    for i, (sp, shape, rshape) in enumerate(zip(spacing, voxel.shape, resized.shape)):
        new_spacing[i] = float(sp) * shape / rshape
    return resized, new_spacing


def rotation(array, angle):
    '''using Euler angles method.
    @author: renchao
    @params:
        angle: 0: no rotation, 1: rotate 90 deg, 2: rotate 180 deg, 3: rotate 270 deg
    '''
    #
    X = np.rot90(array, angle[0], axes=(0, 1))  # rotate in X-axis
    Y = np.rot90(X, angle[1], axes=(0, 2))  # rotate in Y'-axis
    Z = np.rot90(Y, angle[2], axes=(1, 2))  # rotate in Z"-axis
    return Z


def reflection(array, axis):
    '''
    @author: renchao
    @params:
        axis: -1: no flip, 0: Z-axis, 1: Y-axis, 2: X-axis
    '''
    if axis != -1:
        ref = np.flip(array, axis)
    else:
        ref = np.copy(array)
    return ref


def crop(array, zyx, dhw):
    z, y, x = zyx
    d, h, w = dhw
    cropped = array[z - d // 2:z + d // 2,
              y - h // 2:y + h // 2,
              x - w // 2:x + w // 2]
    return cropped


def random_center(shape, move):
    offset = np.random.randint(-move, move + 1, size=3)
    zyx = np.array(shape) // 2 + offset
    return zyx
