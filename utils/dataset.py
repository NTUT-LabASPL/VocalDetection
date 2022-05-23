import os
import tensorflow as tf
import h5py

from definitions import DATASET_DIR


def load(ds_path):
    """
    ds_path(str) ==> h5 dataset path
    >>> load(file_path)
    tf.data.Dataset
    """
    h5_path = os.path.join(DATASET_DIR, ds_path)
    file = h5py.File(h5_path, 'r')
    x = file['X']
    y = file['Y']
    return tf.data.Dataset.from_tensor_slices((x, y))


def create_new_dataset_from_index(ds_path, indices):
    indices = sorted(indices)
    h5_path = os.path.join(DATASET_DIR, ds_path)
    file = h5py.File(h5_path, 'r')
    x = file['X']
    y = file['Y']
    return tf.data.Dataset.from_tensor_slices((x[indices], y[indices]))


def get_dataset_without_label(ds_path):
    h5_path = os.path.join(DATASET_DIR, ds_path)
    file = h5py.File(h5_path, 'r')
    return file['X']

def get_ground_truth(ds_path):
    h5_path = os.path.join(DATASET_DIR, ds_path)
    file = h5py.File(h5_path, 'r')
    return file['Y']
