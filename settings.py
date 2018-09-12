import os

DATA_DIR = r'D:\data\salt'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

TRAIN_IMG_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_MASK_DIR =  os.path.join(TRAIN_DIR, 'masks')
TEST_IMG_DIR = os.path.join(TEST_DIR, 'images')

LABEL_FILE = os.path.join(DATA_DIR, 'train.csv')
DEPTHS_FILE = os.path.join(DATA_DIR, 'depths.csv')
META_FILE = os.path.join(DATA_DIR, 'meta.csv')

MODEL_DIR = 'models'

ID_COLUMN = 'id'
DEPTH_COLUMN = 'z'
X_COLUMN = 'file_path_image'
Y_COLUMN = 'file_path_mask'

H = W = 128
ORIG_H = ORIG_W = 101