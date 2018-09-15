import os
import pandas as pd
import torch
import torch.nn as nn
import settings
import utils
from unet_models import UNetResNet

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
    convert_model2()
    #get_mask_existence()