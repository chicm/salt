import os
import json
import pathlib
import random
import sys
import time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as cocomask
from sklearn.model_selection import BaseCrossValidator, KFold

import settings

def create_submission(meta, predictions):
    output = []
    for image_id, mask in zip(meta['id'].values, predictions):
        rle_encoded = ' '.join(str(rle) for rle in run_length_encoding(mask))
        output.append([image_id, rle_encoded])

    submission = pd.DataFrame(output, columns=['id', 'rle_mask']).astype(str)
    return submission


def encode_rle(predictions):
    return [run_length_encoding(mask) for mask in predictions]


def read_masks(img_ids):
    masks = []
    for img_id in img_ids:
        base_filename = '{}.png'.format(img_id)
        mask = Image.open(os.path.join(settings.TRAIN_MASK_DIR, base_filename))
        mask = np.asarray(mask.convert('L').point(lambda x: 0 if x < 128 else 1)).astype(np.uint8)
        masks.append(mask)
    return masks


def run_length_encoding(x):
    # https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b > prev + 1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b
    return rle


def run_length_decoding(mask_rle, shape):
    """
    Based on https://www.kaggle.com/msl23518/visualize-the-stage1-test-solution and modified
    Args:
        mask_rle: run-length as string formatted (start length)
        shape: (height, width) of array to return

    Returns:
        numpy array, 1 - mask, 0 - background

    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[1] * shape[0], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 255
    return img.reshape((shape[1], shape[0])).T

def get_salt_existence():
    train_mask = pd.read_csv(settings.LABEL_FILE)
    salt_exists_dict = {}
    for row in train_mask.values:
        #print(row[1] is np.nan)
        salt_exists_dict[row[0]] = 0 if (row[1] is np.nan or len(row[1]) < 1) else 1
    return salt_exists_dict

def generate_metadata(train_images_dir, test_images_dir, depths_filepath):
    depths = pd.read_csv(depths_filepath)
    salt_exists_dict = get_salt_existence()

    metadata = {}
    for filename in tqdm(os.listdir(os.path.join(train_images_dir, 'images'))):
        image_filepath = os.path.join(train_images_dir, 'images', filename)
        mask_filepath = os.path.join(train_images_dir, 'masks', filename)
        image_id = filename.split('.')[0]
        depth = depths[depths['id'] == image_id]['z'].values[0]

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(mask_filepath)
        metadata.setdefault('is_train', []).append(1)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('z', []).append(depth)
        metadata.setdefault('salt_exists', []).append(salt_exists_dict[image_id])

    for filename in tqdm(os.listdir(os.path.join(test_images_dir, 'images'))):
        image_filepath = os.path.join(test_images_dir, 'images', filename)
        image_id = filename.split('.')[0]
        depth = depths[depths['id'] == image_id]['z'].values[0]

        metadata.setdefault('file_path_image', []).append(image_filepath)
        metadata.setdefault('file_path_mask', []).append(None)
        metadata.setdefault('is_train', []).append(0)
        metadata.setdefault('id', []).append(image_id)
        metadata.setdefault('z', []).append(depth)
        metadata.setdefault('salt_exists', []).append(0)

    return pd.DataFrame(metadata)

def rle_from_binary(prediction):
    prediction = np.asfortranarray(prediction)
    return cocomask.encode(prediction)


def binary_from_rle(rle):
    return cocomask.decode(rle)


def get_segmentations(labeled):
    nr_true = labeled.max()
    segmentations = []
    for i in range(1, nr_true + 1):
        msk = labeled == i
        segmentation = rle_from_binary(msk.astype('uint8'))
        segmentation['counts'] = segmentation['counts'].decode("UTF-8")
        segmentations.append(segmentation)
    return segmentations


def get_crop_pad_sequence(vertical, horizontal):
    top = int(vertical / 2)
    bottom = vertical - top
    right = int(horizontal / 2)
    left = horizontal - right
    return (top, right, bottom, left)


class KFoldBySortedValue(BaseCrossValidator):
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _iter_test_indices(self, X, y=None, groups=None):
        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        sorted_idx_vals = sorted(zip(indices, X), key=lambda x: x[1])
        indices = [idx for idx, val in sorted_idx_vals]

        for split_start in range(self.n_splits):
            split_indeces = indices[split_start::self.n_splits]
            yield split_indeces

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

def get_train_split():
    meta = pd.read_csv(settings.META_FILE, na_filter=False)
    meta_train = meta[meta['is_train'] == 1]
    print(meta.head())

    cv = KFoldBySortedValue()
    for train_idx, valid_idx in cv.split(meta_train[settings.DEPTH_COLUMN].values.reshape(-1)):
        print(len(train_idx), len(valid_idx))
        #break

    meta_train_split, meta_valid_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]
    return meta_train_split, meta_valid_split

def get_nfold_split(ifold, nfold=10, meta_version=1):
    if meta_version == 2:
        return get_nfold_split2(ifold, nfold)

    meta = pd.read_csv(settings.META_FILE, na_filter=False)
    meta_train = meta[meta['is_train'] == 1]

    kf = KFold(n_splits=nfold)
    for i, (train_index, valid_index) in enumerate(kf.split(meta_train[settings.ID_COLUMN].values.reshape(-1))):
        if i == ifold:
            break
    #print(train_index[:10], train_index[-10:])
    #print(valid_index[:10], valid_index[-10:])

    return meta_train.iloc[train_index], meta_train.iloc[valid_index]

def get_nfold_split2(ifold, nfold=10):
    meta_train = pd.read_csv(os.path.join(settings.DATA_DIR, 'train_meta2.csv'))

    with open(os.path.join(settings.DATA_DIR, 'train_split.json'), 'r') as f:
        train_splits = json.load(f)
    train_index = train_splits[str(ifold)]['train_index']
    valid_index = train_splits[str(ifold)]['val_index']
    #print(train_index[:10], train_index[-10:])
    #print(valid_index[:10], valid_index[-10:])
    #print(meta_train.iloc[train_index].head())

    return meta_train.iloc[train_index], meta_train.iloc[valid_index] 


def get_test_meta():
    meta = pd.read_csv(settings.META_FILE, na_filter=False)
    test_meta = meta[meta['is_train'] == 0]
    print(len(test_meta.values))
    return test_meta

if __name__ == '__main__':
    #get_test_meta()
    get_nfold_split(2)
