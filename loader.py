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
    def __init__(self, train_mode, meta):
        self.augment_with_target = ImgAug(aug.crop_seq(crop_size=(H, W), pad_size=(32,32), pad_method='reflect'))
        self.image_augment = ImgAug()
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = None #transforms.Compose([transforms.ToTensor()])

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
        #img = self.transform(img)

        if self.train_mode:
            mask_fn = self.mask_filenames[index]
            mask = self.load_image(mask_fn, True)
            #mask = self.transform(mask)
            #img, mask = self.img_augment(img, mask)
            return img, mask
        else:
            return [img]

    def aug_image(self, img, mask):
        if mask is not None:
            Mi = from_pil(mask)
            Mi = [to_pil(Mi == class_nr) for class_nr in [0, 1]]
            
            Xi, *Mi = from_pil(img, *Mi)
            Xi, *Mi = self.augment_with_target(Xi, *Mi)

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

def get_train_loaders(batch_size=8):
    train_meta, val_meta = get_train_split()
    print(train_meta[X_COLUMN].values[:5])
    print(train_meta[Y_COLUMN].values[:5])

    train_set = ImageDataset(True, train_meta)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=train_set.collate_fn)
    train_loader.num = len(train_set)

    val_set = ImageDataset(True, val_meta)
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

if __name__ == '__main__':
    #test_test_loader()
    test_train_loader()
    #small_dict, img_ids = load_small_train_ids()
    #print(img_ids[:10])
