import pandas as pd
import settings
import utils

def prepare_metadata():
    print('creating metadata')
    meta = utils.generate_metadata(train_images_dir=settings.TRAIN_DIR,
                                   test_images_dir=settings.TEST_DIR,
                                   depths_filepath=settings.DEPTHS_FILE
                                   )
    meta.to_csv(settings.META_FILE, index=None)

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
    test()