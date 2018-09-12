import argparse
import logging as log
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import pdb
import settings
from loader import get_train_loaders
from unet_models import UNetResNet
from lovasz_losses import lovasz_hinge
from postprocessing import crop_image, binarize
from metrics import intersection_over_union, intersection_over_union_thresholds

epochs = 200
batch_size = 16
CKP = 'models/152/best_814_elu.pth'

def train(args):
    print('start training...')
    model = UNetResNet(152, 2, pretrained=True, is_deconv=True)
    #model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    criterion = lovasz_hinge 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, val_loader = get_train_loaders(args.ifold, batch_size=batch_size, dev_mode=False)
    #validate(model, val_loader, criterion)
    lr_scheduler = CosineAnnealingLR(optimizer, 15, 1e-7) #ExponentialLR(optimizer, 0.9, last_epoch=-1)

    best_iout = 0

    for epoch in range(epochs):
        lr_scheduler.step()
        train_loss = 0
        model.train()
        model.freeze_bn()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'lr:{current_lr}')
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            img, target = data
            img, target = img.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print(f'epoch {epoch}: {batch_size*(batch_idx+1)}/{train_loader.num} batch loss: {loss.item() : .4f}, avg loss: {train_loss/(batch_idx+1) : .4f} lr: {current_lr: .7f}', end='\r')
        print('\n')
        print(f'epoch {epoch}: {(time.time() - bg) / 60: .2f} minutes')

        iout, iou, val_loss = validate(model, val_loader, criterion)

        if iout > best_iout:
            best_iout = iout
            torch.save(model.state_dict(), f'models/152/best_{args.ifold}.pth')

        log.info(f'epoch {epoch}: train loss: {train_loss: .4f} val loss: {val_loss: .4f} iout: {iout: .4f} best iout: {best_iout: .4f} iou: {iou: .4f} lr: {current_lr: .7f}')
        

def validate(model, val_loader, criterion):
    model.eval()
    print('validating...')
    outputs = []
    val_loss = 0
    with torch.no_grad():
        for img, target in val_loader:
            img, target = img.cuda(), target.cuda()
            output = model(img)

            loss = criterion(output, target)
            val_loss += loss.item()
            output = torch.sigmoid(output)
            for o in output.cpu().numpy():
                outputs.append(o)

    n_batches = val_loader.num // batch_size if val_loader.num % batch_size == 0 else val_loader.num // batch_size + 1

    # y_pred, list of 400 np array, each np array's shape is 101,101
    y_pred = generate_preds(outputs, (settings.ORIG_H, settings.ORIG_W))
    print(f'Validation loss: {val_loss/n_batches: .4f}')

    iou_score = intersection_over_union(val_loader.y_true, y_pred)
    iout_score = intersection_over_union_thresholds(val_loader.y_true, y_pred)
    print(f'IOU score on validation is {iou_score:.4f}')
    print(f'IOUT score on validation is {iout_score:.4f}')

    return iout_score, iou_score, val_loss / n_batches

def generate_preds(outputs, target_size):
    preds = []

    for output in outputs:
        cropped = crop_image(output, target_size=target_size)
        pred = binarize(cropped, 0.5)
        preds.append(pred)

    return preds

if __name__ == '__main__':
    
    log.basicConfig(
        filename = 'trainlog.txt', 
        format   = '%(asctime)s : %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S', 
        level = log.INFO)
    #pdb.set_trace()
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--ifold', default=0, type=int, help='kfold index')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    train(args)
