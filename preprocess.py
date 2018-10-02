import os
import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from keras.preprocessing.image import load_img
from sklearn.model_selection import StratifiedKFold
import settings
import utils
from unet_models import UNetResNet

DATA_DIR = settings.DATA_DIR

def prepare_metadata():
    print('creating metadata')
    meta = utils.generate_metadata(train_images_dir=settings.TRAIN_DIR,
                                   test_images_dir=settings.TEST_DIR,
                                   depths_filepath=settings.DEPTHS_FILE
                                   )
    meta.to_csv(settings.META_FILE, index=None)

def convert_model():
    model = UNetResNet(152, 2, pretrained=True, is_deconv=True)
    model = nn.DataParallel(model)
    old_model_file = os.path.join(settings.MODEL_DIR, 'old', '152', 'best_814_elu.pth')
    new_model_file = os.path.join(settings.MODEL_DIR, '152', 'best_814_elu.pth')
    
    print('loading... {}'.format(old_model_file))
    model.load_state_dict(torch.load(old_model_file))
    
    print('saving... {}'.format(new_model_file))
    torch.save(model.module.state_dict(), new_model_file)

def convert_model2():
    model = UNetResNet(152, 2, pretrained=True, is_deconv=True)
    model.classifier = None
    old_model_file = os.path.join(settings.MODEL_DIR, '152', 'best_814_elu.pth')
    model.load_state_dict(torch.load(old_model_file))

    model.final = nn.Conv2d(32, 1, kernel_size=1)
    model.classifier =  nn.Linear(32 * 256 * 256, 1)

    new_model_file = os.path.join(settings.MODEL_DIR, '152_new', 'best_814.pth')
    torch.save(model.state_dict(), new_model_file)

def cov_to_class(val):    
    for i in range(0, 11):
        if val * 10 <= i :
            return i

def generate_stratified_metadata():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), index_col="id", usecols=[0])
    depths_df = pd.read_csv(os.path.join(DATA_DIR, "depths.csv"), index_col="id")
    train_df = train_df.join(depths_df)
    train_df["masks"] = [np.array(load_img(os.path.join(DATA_DIR, "train", "masks", "{}.png".format(idx)), grayscale=True)) / 255 for idx in train_df.index]
    train_df["coverage"] = train_df.masks.map(np.sum) / pow(settings.ORIG_H, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    train_df["salt_exists"] = train_df.coverage_class.map(lambda x: 0 if x == 0 else 1)
    train_df["is_train"] = 1
    train_df["file_path_image"] = train_df.index.map(lambda x: os.path.join(settings.TRAIN_IMG_DIR, '{}.png'.format(x)))
    train_df["file_path_mask"] = train_df.index.map(lambda x: os.path.join(settings.TRAIN_MASK_DIR, '{}.png'.format(x)))

    train_df.to_csv(os.path.join(settings.DATA_DIR, 'train_meta2.csv'), 
        columns=['file_path_image','file_path_mask','is_train','z','salt_exists', 'coverage_class', 'coverage'])
    train_splits = {}

    kf = StratifiedKFold(n_splits=10)
    for i, (train_index, valid_index) in enumerate(kf.split(train_df.index.values.reshape(-1), train_df.coverage_class.values.reshape(-1))):
        train_splits[str(i)] = {
            'train_index': train_index.tolist(),
            'val_index': valid_index.tolist()  
        }
    with open(os.path.join(settings.DATA_DIR, 'train_split.json'), 'w') as f:
        json.dump(train_splits, f, indent=4)


def test():
    meta = pd.read_csv(settings.META_FILE)
    meta_train = meta[meta['is_train'] == 1]
    print(type(meta_train))

    cv = utils.KFoldBySortedValue()
    for train_idx, valid_idx in cv.split(meta_train[settings.DEPTH_COLUMN].values.reshape(-1)):
        print(len(train_idx), len(valid_idx))
        print(train_idx[:10])
        print(valid_idx[:10])
        #break

    meta_train_split, meta_valid_split = meta_train.iloc[train_idx], meta_train.iloc[valid_idx]
    print(type(meta_train_split))
    print(meta_train_split[settings.X_COLUMN].values[:10])

if __name__ == '__main__':
    #prepare_metadata()
    #test()
    #convert_model2()
    #get_mask_existence()
    generate_stratified_metadata()
    #get_nfold_split2(0)