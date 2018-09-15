import os
import argparse
import logging as log
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, _LRScheduler, ReduceLROnPlateau
import pdb
import settings
from loader import get_train_loaders
from unet_models import UNetResNet
from lovasz_losses import lovasz_hinge
from postprocessing import crop_image, binarize
from metrics import intersection_over_union, intersection_over_union_thresholds

epochs = 80
batch_size = 64
MODEL_DIR = settings.MODEL_DIR
#CKP = '{}/152/best_814_elu.pth'.format(MODEL_DIR)

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

def train(args):
    print('start training...')
    model_file = '{}/34/best_{}.pth'.format(MODEL_DIR, args.ifold)

    model = UNetResNet(34, 2, pretrained=True, is_deconv=True)
    CKP = model_file
    if os.path.exists(CKP):
        print('loading {}...'.format(CKP))
        model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    criterion = lovasz_hinge 
    optimizer = optim.Adam(model.parameters(), lr=args.lr) #, weight_decay=0.0001)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    train_loader, val_loader = get_train_loaders(args.ifold, batch_size=batch_size, dev_mode=False)

    # CyclicExponentialLR(optimizer, 0.9, init_lr=args.lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    #CyclicExponentialLR(optimizer, 0.8, init_lr=args.lr) #ExponentialLR(optimizer, 0.9, last_epoch=-1) #CosineAnnealingLR(optimizer, 15, 1e-7) 

    best_iout, _, _ = validate(model, val_loader, criterion)
    model.train()
    lr_scheduler.step(best_iout)

    for epoch in range(args.start_epoch, epochs):
        train_loss = 0

        #if epoch < 5:
        #    model.freeze_bn()
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

            # adamW
            wd = 0.0001
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-wd * group['lr'], param.data)

            optimizer.step()

            train_loss += loss.item()
            print('epoch {}: {}/{} batch loss: {:.4f}, avg loss: {:.4f} lr: {:.7f}'
                .format(epoch, batch_size*(batch_idx+1), train_loader.num, loss.item(), train_loss/(batch_idx+1), current_lr), end='\r')
        print('\n')
        print('epoch {}: {:.2f} minutes'.format(epoch, (time.time() - bg) / 60))

        iout, iou, val_loss = validate(model, val_loader, criterion)

        if iout > best_iout:
            best_iout = iout
            print('saving {}...'.format(model_file))
            torch.save(model.state_dict(), model_file)

        log.info('epoch {}: train loss: {:.4f} val loss: {:.4f} iout: {:.4f} best iout: {:.4f} iou: {:.4f} lr: {:.7f}'
            .format(epoch, train_loss, val_loss, iout, best_iout, iou, current_lr))

        model.train()
        lr_scheduler.step(best_iout)
        

def validate(model, val_loader, criterion, threshold=0.5):
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
    y_pred = generate_preds(outputs, (settings.ORIG_H, settings.ORIG_W), threshold)
    print('Validation loss: {:.4f}'.format(val_loss/n_batches))

    iou_score = intersection_over_union(val_loader.y_true, y_pred)
    iout_score = intersection_over_union_thresholds(val_loader.y_true, y_pred)
    print('IOU score on validation is {:.4f}'.format(iou_score))
    print('IOUT score on validation is {:.4f}'.format(iout_score))

    return iout_score, iou_score, val_loss / n_batches

def find_threshold():
    ckp = r'G:\salt\models\152\ensemble_822\best_3.pth'
    model = UNetResNet(152, 2, pretrained=True, is_deconv=True)
    model.load_state_dict(torch.load(ckp))
    model = model.cuda()
    criterion = lovasz_hinge
    _, val_loader = get_train_loaders(3, batch_size=batch_size, dev_mode=False)

    best, bestt = 0, 0.
    for t in range(35, 55, 1):
        iout, _, _ = validate(model, val_loader, criterion, t/100.)
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

if __name__ == '__main__':
    
    log.basicConfig(
        filename = 'trainlog.txt', 
        format   = '%(asctime)s : %(message)s',
        datefmt  = '%Y-%m-%d %H:%M:%S', 
        level = log.INFO)
    #pdb.set_trace()
    parser = argparse.ArgumentParser(description='Salt segmentation')
    parser.add_argument('--lr', default=0.00002, type=float, help='learning rate')
    parser.add_argument('--ifold', default=0, type=int, help='kfold index')
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()

    #find_threshold()
    for i in range(3):
        args.ifold = i
        train(args)
