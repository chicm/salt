import os, cv2, glob
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torchvision import datasets, models, transforms
from settings import *
from utils import get_train_split, ImgAug, from_pil, to_pil
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
    
        self.img_ids = meta[ID_COLUMN].values
        self.img_filenames = meta[X_COLUMN].values
        
        if self.train_mode:
            self.mask_filenames = meta[Y_COLUMN].values
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_fn = self.img_filenames[index]
        img = self.load_image(img_fn)

        if self.train_mode:
            mask_fn = self.mask_filenames[index]
            mask = self.load_image(mask_fn, True)
            img, mask = self.aug_image(img, mask)
            return img, mask
        else:
            img = self.aug_image(img)
            return [img]

    def aug_image(self, img, mask):
        if mask is not None:
            Mi = from_pil(mask)
            Mi = [to_pil(Mi == class_nr) for class_nr in [0, 1]]
            
            Xi, *Mi = from_pil(img, *Mi)
            Xi, *Mi = self.augment_with_target(Xi, *Mi)
            Xi = self.image_augment(Xi)
            Xi, *Mi = to_pil(Xi, *Mi)

            if self.mask_transform is not None:
                Mi = [self.mask_transform(m) for m in Mi]

            if self.image_transform is not None:
                Xi = self.image_transform(Xi)

            return Xi, torch.cat(Mi, dim=0)
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
            image = image.convert('L').point(lambda x: 0 if x < 128 else 255, '1')
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
            return inputs, labels
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

def get_train_loaders(batch_size=8):
    train_meta, val_meta = get_train_split()
    print(train_meta[X_COLUMN].values[:5])
    print(train_meta[Y_COLUMN].values[:5])

    image_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225]),
                                        ])
    mask_transform = transforms.Compose([transforms.Lambda(to_array),
                                         transforms.Lambda(to_tensor),
                                        ])

    train_set = ImageDataset(True, train_meta,
                            augment_with_target=ImgAug(aug.crop_seq(crop_size=(H, W), pad_size=(32,32), pad_method='reflect')),
                            image_augment=ImgAug(aug.intensity_seq),
                            image_transform=image_transform,
                            mask_transform=mask_transform)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)
    train_loader.num = len(train_set)

    val_set = ImageDataset(True, val_meta,
                            augment_with_target=ImgAug(aug.pad_to_fit_net(64, 'reflect')),
                            image_augment=ImgAug(aug.pad_to_fit_net(64, 'reflect')),
                            image_transform=image_transform,
                            mask_transform=mask_transform)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=val_set.collate_fn)
    val_loader.num = len(val_set)

    return train_loader, val_loader

def test_train_loader():
    train_loader, val_loader = get_train_loaders(4)
    print(train_loader.num, val_loader.num)
    for i, data in enumerate(train_loader):
        imgs, masks = data
        #pdb.set_trace()
        print(imgs.size(), masks.size())
        print(imgs)
        print(masks)
        if i > 5:
            break

if __name__ == '__main__':
    #test_test_loader()
    test_train_loader()
    #small_dict, img_ids = load_small_train_ids()
    #print(img_ids[:10])