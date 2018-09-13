import os
import argparse
import logging as log
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler
import pdb
import settings
from loader import get_train_loaders
from unet_models import UNetResNet
from lovasz_losses import lovasz_hinge
from postprocessing import crop_image, binarize
from metrics import intersection_over_union, intersection_over_union_thresholds

epochs = 200
batch_size = 8
MODEL_DIR = settings.MODEL_DIR
CKP = 'models/152/best_814_elu.pth'

class CyclicExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma, init_lr=0.00015, min_lr=1e-7, restart_max_lr=1e-5, last_epoch=-1):
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

def train(args):
    print('start training...')
    model_file = '{}/152/best_{}.pth'.format(MODEL_DIR, args.ifold)

    model = UNetResNet(152, 2, pretrained=True, is_deconv=True)
    if os.path.exists(model_file):
        print('loading {}...'.format(model_file))
        model.load_state_dict(torch.load(model_file))
    model = model.cuda()

    criterion = lovasz_hinge 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)

    train_loader, val_loader = get_train_loaders(args.ifold, batch_size=batch_size, dev_mode=False)
    #validate(model, val_loader, criterion)
    # CyclicExponentialLR(optimizer, 0.9, init_lr=args.lr)
    lr_scheduler = CyclicExponentialLR(optimizer, 0.9, init_lr=args.lr) #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    best_iout = 0

    for epoch in range(args.start_epoch, epochs):
        lr_scheduler.step()
        train_loss = 0
        model.train()
        if epoch < 5:
            model.freeze_bn()
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print('lr:', current_lr)
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
            print('epoch {}: {}/{} batch loss: {:.4f}, avg loss: {:.4f} lr: {:.7f}'
                .format(epoch, batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1), current_lr), end='\r')
        print('\n')
        print('epoch {}: {:.2f} minutes'.format(epoch, (time.time() - bg) / 60))

        iout, iou, val_loss = validate(model, val_loader, criterion)

        if iout > best_iout:
            best_iout = iout
            torch.save(model.state_dict(), model_file)

        log.info('epoch {}: train loss: {:.4f} val loss: {:.4f} iout: {:.4f} best iout: {:.4f} iou: {:.4f} lr: {:.7f}'
            .format(epoch, train_loss, val_loss, iout, best_iout, iou, current_lr))
        

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
    print('Validation loss: {:.4f}'.format(val_loss/n_batches))

    iou_score = intersection_over_union(val_loader.y_true, y_pred)
    iout_score = intersection_over_union_thresholds(val_loader.y_true, y_pred)
    print('IOU score on validation is {:.4f}'.format(iou_score))
    print('IOUT score on validation is {:.4f}'.format(iout_score))

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
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    train(args)
