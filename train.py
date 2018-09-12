import torch
import torch.optim as optim
import torch.nn.functional as F

import settings
from loader import get_train_loaders
from unet_models import UNetResNet
from lovasz_losses import lovasz_hinge
from postprocessing import crop_image, binarize
from metrics import intersection_over_union, intersection_over_union_thresholds

batch_size = 8
CKP = 'models/152/best_814_elu.pth'

def train():
    model = UNetResNet(152, 2, pretrained=True, is_deconv=True)
    model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    criterion = lovasz_hinge 
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    train_loader, val_loader = get_train_loaders(batch_size, dev_mode=False)
    validate(model, val_loader, criterion)

    for epoch in range(10):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            img, mask = data
            img, mask = img.cuda(), mask.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            print(f'epoch: {epoch} {batch_size*(batch_idx+1)}/{train_loader.num} batch loss: {loss.item() : .4f}, avg loss: {train_loss/(batch_idx+1) : .4f}', end='\r')
        print('\n')

        validate(model, val_loader, criterion)

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
    model.train()

    n_batches = val_loader.num // batch_size if val_loader.num % batch_size == 0 else val_loader.num // batch_size + 1

    y_pred = generate_preds(outputs, (settings.ORIG_H, settings.ORIG_W))
    print(f'Validation loss: {val_loss/n_batches: .4f}')
    # y_true, list of 400 np array, each np array's shape is 101,101
    iou_score = intersection_over_union(val_loader.y_true, y_pred)
    iout_score = intersection_over_union_thresholds(val_loader.y_true, y_pred)
    print(f'IOU score on validation is {iou_score:.4f}')
    print(f'IOUT score on validation is {iout_score:.4f}')

def generate_preds(outputs, target_size):
    preds = []

    for output in outputs:
        cropped = crop_image(output, target_size=target_size)
        pred = binarize(cropped, 0.5)
        preds.append(pred)

    return preds

if __name__ == '__main__':
    train()
