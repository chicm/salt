import os
import argparse
import logging as log
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import pdb
import settings
from loader import get_train_loaders
from unet_models import UNetResNet, UNetResNetAtt, UNetResNetV3
from unet_new import UNetResNetV4
from unet_se import UNetResNetSE
from lovasz_losses import lovasz_hinge, lovasz_softmax
from dice_losses import mixed_dice_bce_loss, FocalLoss2d
from postprocessing import crop_image, binarize, crop_image_softmax
from metrics import intersection_over_union, intersection_over_union_thresholds

MODEL_DIR = settings.MODEL_DIR
focal_loss2d = FocalLoss2d()

class CyclicExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, init_lr, min_lr=5e-7, restart_max_lr=1e-5, last_epoch=-1):
        self.gamma = gamma
        self.last_lr = init_lr
        self.min_lr = min_lr
        self.restart_max_lr = restart_max_lr
        super(CyclicExponentialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = self.last_lr * self.gamma
        if lr < self.min_lr:
            lr = self.restart_max_lr
        self.last_lr = lr
        return [lr]*len(self.base_lrs)

def weighted_loss(output, target, epoch=0):
    mask_output, _ = output
    mask_target, _ = target
    
    lovasz_loss = lovasz_hinge(mask_output, mask_target)
    #dice_loss = mixed_dice_bce_loss(mask_output, mask_target)
    focal_loss = focal_loss2d(mask_output, mask_target)
    if epoch < 10:
        return focal_loss
    else:
        return lovasz_loss #, lovasz_loss.item(), bce_loss.item()

def train(args):
    print('start training...')

    model = UNetResNetV4(args.layers)
    if args.exp_name is None:
        model_file = os.path.join(MODEL_DIR, model.name, 'best_{}.pth'.format(args.ifold))
    else:
        model_file = os.path.join(MODEL_DIR, args.exp_name, model.name, 'best_{}.pth'.format(args.ifold))

    parent_dir = os.path.dirname(model_file)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    CKP = model_file
    if os.path.exists(CKP):
        print('loading {}...'.format(CKP))
        model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    train_loader, val_loader = get_train_loaders(args.ifold, batch_size=args.batch_size, dev_mode=False, pad_mode=args.pad_mode)

    if args.lrs == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=args.factor, patience=args.patience, min_lr=args.min_lr)
    else:
        lr_scheduler = CosineAnnealingLR(optimizer, args.t_max, eta_min=args.min_lr)
    #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    best_iout, _, _ = validate(args, model, val_loader, args.start_epoch)
    model.train()

    if args.lrs == 'plateau':
        lr_scheduler.step(best_iout)
    else:
        lr_scheduler.step()

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = 0

        #if epoch < 5:
        #    model.freeze_bn()
        current_lr = get_lrs(optimizer)  #optimizer.state_dict()['param_groups'][2]['lr']
        print('lr:', current_lr)
        bg = time.time()
        for batch_idx, data in enumerate(train_loader):
            img, target, salt_target = data
            #add_depth_channel(img)
            img, target, salt_target = img.cuda(), target.cuda(), salt_target.cuda()
            optimizer.zero_grad()
            output, salt_out = model(img)
            
            loss = weighted_loss((output, salt_out), (target, salt_target), epoch=epoch)
            loss.backward()
 
            # adamW
            #wd = 0.0001
            #for group in optimizer.param_groups:
            #    for param in group['params']:
            #        param.data = param.data.add(-wd * group['lr'], param.data)

            optimizer.step()

            train_loss += loss.item()
            print('epoch {}: {}/{} batch loss: {:.4f}, avg loss: {:.4f} lr: {}'
                .format(epoch, args.batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1), current_lr), end='\r')
        print('\n')
        print('epoch {}: {:.2f} minutes'.format(epoch, (time.time() - bg) / 60))

        iout, iou, val_loss = validate(args, model, val_loader, epoch=epoch)

        if iout > best_iout:
            best_iout = iout
            print('saving {}...'.format(model_file))
            torch.save(model.state_dict(), model_file)

        log.info('epoch {}: train loss: {:.4f} val loss: {:.4f} iout: {:.4f} best iout: {:.4f} iou: {:.4f} lr: {}'
            .format(epoch, train_loss, val_loss, iout, best_iout, iou, current_lr))

        model.train()
        
        if args.lrs == 'plateau':
            lr_scheduler.step(best_iout)
        else:
            lr_scheduler.step()

    del model, train_loader, val_loader, optimizer, lr_scheduler
        
def get_lrs(optimizer):
    lrs = []
    for pgs in optimizer.state_dict()['param_groups']:
        lrs.append(pgs['lr'])
    lrs = ['{:.6f}'.format(x) for x in lrs]
    return lrs

def validate(args, model, val_loader, epoch=0, threshold=0.5):
    model.eval()
    print('validating...')
    outputs = []
    val_loss = 0
    with torch.no_grad():
        for img, target, salt_target in val_loader:
            #add_depth_channel(img)
            img, target, salt_target = img.cuda(), target.cuda(), salt_target.cuda()
            output, salt_out = model(img)
            #print(output.size(), salt_out.size())

            loss = weighted_loss((output, salt_out), (target, salt_target), epoch=epoch)
            val_loss += loss.item()
            output = torch.sigmoid(output)
            
            for o in output.cpu():
                outputs.append(o.squeeze().numpy())

    n_batches = val_loader.num // args.batch_size if val_loader.num % args.batch_size == 0 else val_loader.num // args.batch_size + 1

    # y_pred, list of 400 np array, each np array's shape is 101,101
    y_pred = generate_preds_softmax(outputs, (settings.ORIG_H, settings.ORIG_W), threshold)
    print(y_pred[0].shape)
    print('Validation loss: {:.4f}'.format(val_loss/n_batches))

    iou_score = intersection_over_union(val_loader.y_true, y_pred)
    iout_score = intersection_over_union_thresholds(val_loader.y_true, y_pred)
    print('IOU score on validation is {:.4f}'.format(iou_score))
    print('IOUT score on validation is {:.4f}'.format(iout_score))

    return iout_score, iou_score, val_loss / n_batches

def find_threshold(args):
    #ckp = r'G:\salt\models\152\ensemble_822\best_3.pth'
    ckp = r'D:\data\salt\models\UNetResNetV4_34\best_0.pth'
    model = UNetResNetV4(34)
    model.load_state_dict(torch.load(ckp))
    model = model.cuda()
    #criterion = lovasz_hinge
    _, val_loader = get_train_loaders(0, batch_size=args.batch_size, dev_mode=False)

    best, bestt = 0, 0.
    for t in range(40, 55, 1):
        print('threshold:', t/100.)
        iout, _, _ = validate(args, model, val_loader, epoch=10, threshold=t/100.)
        if iout > best:
            best = iout
            bestt = t/100.
    print('best:', best, bestt)

def generate_preds(outputs, target_size, threshold=0.5):
    preds = []

    for output in outputs:
        cropped = crop_image(output, target_size=target_size)
        pred = binarize(cropped, threshold)
        preds.append(pred)

    return preds

def generate_preds_softmax(outputs, target_size, threshold=0.5):
    preds = []

    for output in outputs:
        cropped = crop_image_softmax(output, target_size=target_size)
        pred = binarize(cropped, threshold)
        preds.append(pred)

    return preds

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--layers', default=34, type=int, help='model layers')
    parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
    parser.add_argument('--min_lr', default=0.0002, type=float, help='min learning rate')
    parser.add_argument('--ifolds', default='0', type=str, help='kfold indices')
    parser.add_argument('--batch_size', default=32, type=int, help='batch_size')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--epochs', default=200, type=int, help='epoch')
    parser.add_argument('--optim', default='SGD', choices=['SGD', 'Adam'], help='optimizer')
    parser.add_argument('--lrs', default='cosine', choices=['cosine', 'plateau'], help='LR sceduler')
    parser.add_argument('--patience', default=6, type=int, help='lr scheduler patience')
    parser.add_argument('--factor', default=0.5, type=float, help='lr scheduler factor')
    parser.add_argument('--t_max', default=8, type=int, help='lr scheduler patience')
    parser.add_argument('--pad_mode', default='edge', choices=['reflect', 'edge'], help='pad method')
    parser.add_argument('--exp_name', default='pre_depths', type=str, help='exp name')
    args = parser.parse_args()

    print(args)
    ifolds = [int(x) for x in args.ifolds.split(',')]
    print(ifolds)
    log.basicConfig(
        filename = 'trainlog_{}.txt'.format(''.join([str(x) for x in ifolds])), 
        format   = '%(asctime)s : %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S', 
        level = log.INFO)

    for i in ifolds:
        args.ifold = i
        train(args)
