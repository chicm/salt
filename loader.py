import os, cv2, glob
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from settings import *
from utils import get_train_split, ImgAug, from_pil, to_pil, read_masks, get_test_meta, get_nfold_split
import augmentation as aug

import pdb

class ImageDataset(data.Dataset):
    def __init__(self, train_mode, meta, augment_with_target=None,
                image_augment=None, image_transform=None, mask_transform=None):
        self.augment_with_target = augment_with_target
        self.image_augment = image_augment
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.train_mode = train_mode
        self.meta = meta
    
        self.img_ids = meta[ID_COLUMN].values
        self.img_filenames = meta[X_COLUMN].values
        self.salt_exists = meta['salt_exists'].values
        
        if self.train_mode:
            self.mask_filenames = meta[Y_COLUMN].values
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        base_img_fn = self.img_filenames[index].split('\\')[-1]
        if self.train_mode:
            img_fn = os.path.join(TRAIN_IMG_DIR, base_img_fn)
        else:
            img_fn = os.path.join(TEST_IMG_DIR, base_img_fn)
        img = self.load_image(img_fn)

        if self.train_mode:
            base_mask_fn = self.mask_filenames[index].split('\\')[-1]
            mask_fn = os.path.join(TRAIN_MASK_DIR, base_mask_fn)
            mask = self.load_image(mask_fn, True)
            img, mask = self.aug_image(img, mask)
            return img, mask, self.salt_exists[index]
        else:
            img = self.aug_image(img)
            return [img]

    def aug_image(self, img, mask=None):
        if mask is not None:
            Xi, Mi = from_pil(img, mask)
            #print('>>>', Xi.shape, Mi.shape)
            #print(Mi)
            Xi, Mi = self.augment_with_target(Xi, Mi)
            if self.image_augment is not None:
                Xi = self.image_augment(Xi)
            Xi, Mi = to_pil(Xi, Mi)

            if self.mask_transform is not None:
                Mi = self.mask_transform(Mi)

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)

            return Xi, Mi#torch.cat(Mi, dim=0)
        else:
            Xi = from_pil(img)
            Xi = self.image_augment(Xi)
            Xi = to_pil(Xi)

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)
            return Xi

    def load_image(self, img_filepath, grayscale=False):
        image = Image.open(img_filepath, 'r')
        if not grayscale:
            image = image.convert('RGB')
        else:
            image = image.convert('L').point(lambda x: 0 if x < 128 else 1, 'L')
            #image = np.asarray(image) #.convert('L')
            #print(np.max(image))
            #print(image)
            #pass
        return image

    def __len__(self):
        return len(self.img_ids)

    def collate_fn(self, batch):
        imgs = [x[0] for x in batch]
        #pdb.set_trace()
        inputs = torch.stack(imgs)

        if self.train_mode:
            masks = [x[1] for x in batch]
            labels = torch.stack(masks)

            salt_target = [x[2] for x in batch]
            return inputs, labels, torch.FloatTensor(salt_target)
        else:
            return inputs

def to_array(x):
    x_ = x.convert('L')  # convert image to monochrome
    x_ = np.array(x_)
    x_ = x_.astype(np.float32)
    return x_


def to_tensor(x):
    x_ = np.expand_dims(x, axis=0)
    x_ = torch.from_numpy(x_)
    return x_

img_transforms = [
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

def get_tta_transforms(index):
    tta_transforms = {
        0: [],
        1: [transforms.RandomHorizontalFlip(p=2.)],
        2: [transforms.RandomVerticalFlip(p=2.)],
        3: [transforms.RandomHorizontalFlip(p=2.), transforms.RandomVerticalFlip(p=2.)]
    }
    return transforms.Compose([*(tta_transforms[index]), *img_transforms])

image_transform = transforms.Compose(img_transforms)
mask_transform = transforms.Compose([transforms.Lambda(to_array),
                                         transforms.Lambda(to_tensor),
                                    ])
#import pdb
def get_train_loaders(ifold, batch_size=8, dev_mode=False):
    #pdb.set_trace()
    train_shuffle = True
    train_meta, val_meta = get_nfold_split(ifold, nfold=10)
    if dev_mode:
        train_shuffle = False
        train_meta = train_meta.iloc[:10]
        val_meta = val_meta.iloc[:10]
    print(train_meta[X_COLUMN].values[:5])
    print(train_meta[Y_COLUMN].values[:5])

    train_set = ImageDataset(True, train_meta,
                            augment_with_target=ImgAug(aug.crop_seq(crop_size=(H, W), pad_size=(28,28), pad_method='reflect')),
                            image_augment=ImgAug(aug.brightness_seq),
                            image_transform=image_transform,
                            mask_transform=mask_transform)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, num_workers=4, collate_fn=train_set.collate_fn, drop_last=True)
    train_loader.num = len(train_set)

    val_set = ImageDataset(True, val_meta,
                            augment_with_target=ImgAug(aug.pad_to_fit_net(64, 'reflect')),
                            image_augment=None, #ImgAug(aug.pad_to_fit_net(64, 'reflect')),
                            image_transform=image_transform,
                            mask_transform=mask_transform)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=val_set.collate_fn)
    val_loader.num = len(val_set)
    val_loader.y_true = read_masks(val_meta[Y_COLUMN].values)

    return train_loader, val_loader

def get_test_loader(batch_size=16, index=0, dev_mode=False):
    test_meta = get_test_meta()
    if dev_mode:
        test_meta = test_meta.iloc[:10]
    test_set = ImageDataset(False, test_meta,
                            image_augment=ImgAug(aug.pad_to_fit_net(64, 'reflect')),
                            image_transform=get_tta_transforms(index))
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=test_set.collate_fn, drop_last=False)
    test_loader.num = len(test_set)
    test_loader.meta = test_set.meta

    return test_loader

def test_train_loader():
    train_loader, val_loader = get_train_loaders(0, batch_size=4, dev_mode=True)
    print(train_loader.num, val_loader.num)
    for i, data in enumerate(train_loader):
        imgs, masks, salt_exists = data
        #pdb.set_trace()
        print(imgs.size(), masks.size(), salt_exists.size())
        print(salt_exists)
        #print(imgs)
        #print(masks)

def test_test_loader():
    test_loader = get_test_loader(4)
    print(test_loader.num)
    for i, data in enumerate(test_loader):
        print(data.size())
        if i > 5:
            break

if __name__ == '__main__':
    #test_test_loader()
    #test_train_loader()
    #small_dict, img_ids = load_small_train_ids()
    #print(img_ids[:10])
    print(get_tta_transforms(3))
