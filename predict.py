import torch
import torch.optim as optim
import torch.nn.functional as F

import settings
from loader import get_test_loader
from unet_models import UNetResNet
from postprocessing import crop_image, binarize
from metrics import intersection_over_union, intersection_over_union_thresholds
from utils import create_submission

batch_size = 32
CKP = 'models/152/best_814_elu.pth'

def predict():
    model = UNetResNet(152, 2, pretrained=True, is_deconv=True)
    model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    test_loader = get_test_loader(batch_size)

    model.eval()
    print('predicting...')
    outputs = []
    with torch.no_grad():
        for i, img in enumerate(test_loader):
            img = img.cuda()
            output = torch.sigmoid(model(img))

            for o in output.cpu().numpy():
                outputs.append(o)
            print(f'{batch_size*(i+1)} / {test_loader.num}', end='\r')

    y_pred_test = generate_preds(outputs, (settings.ORIG_H, settings.ORIG_W))

    submission = create_submission(test_loader.meta, y_pred_test)
    submission_filepath = 'sub2.csv'
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')

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
    predict()
