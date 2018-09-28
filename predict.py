import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import settings
from loader import get_test_loader, add_depth_channel
from unet_models import UNetResNet
from unet_new import UNetResNetV4
from postprocessing import crop_image, binarize, crop_image_softmax
from metrics import intersection_over_union, intersection_over_union_thresholds
from utils import create_submission

batch_size = 64

def do_tta_predict(model, tta_num=4):
    '''
    return 18000x128x128 np array
    '''
    model.eval()
    preds = []
    meta = None

    # i is tta index, 0: no change, 1: horizon flip, 2: vertical flip, 3: do both
    for flip_index in range(tta_num):
        print('flip_index:', flip_index)
        test_loader = get_test_loader(batch_size, index=flip_index, dev_mode=False)
        meta = test_loader.meta
        outputs = None
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                add_depth_channel(img)
                img = img.cuda()
                output, _ = model(img)
                output = torch.sigmoid(output)
                if outputs is None:
                    outputs = output.squeeze()
                else:
                    outputs = torch.cat([outputs, output.squeeze()], 0)

                print('{} / {}'.format(batch_size*(i+1), test_loader.num), end='\r')
        outputs = outputs.cpu().numpy()
        # flip back masks
        if flip_index == 1:
            outputs = np.flip(outputs, 2)
        elif flip_index == 2:
            outputs = np.flip(outputs, 1)
        elif flip_index == 3:
            outputs = np.flip(outputs, 2)
            outputs = np.flip(outputs, 1)
        #print(outputs.shape)
        preds.append(outputs)
    return np.mean(preds, 0), meta

def predict():
    model = UNetResNet(152, pretrained=True, is_deconv=True)
    CKP = os.path.join(settings.MODEL_DIR, '152_new', 'best_0.pth')
    print('loading...', CKP)
    model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    pred, meta = do_tta_predict(model)
    print(pred.shape)
    y_pred_test = generate_preds_softmax(pred, (settings.ORIG_H, settings.ORIG_W))

    submission = create_submission(meta, y_pred_test)
    submission_filepath = 'sub_tta_1.csv'
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')


def ensemble(checkpoints):
    model = UNetResNetV4(34)

    preds = []
    meta = None
    for checkpoint in checkpoints:
        model.load_state_dict(torch.load(checkpoint))
        model = model.cuda()
        print('predicting...', checkpoint)

        pred, meta = do_tta_predict(model, tta_num=2)
        preds.append(pred)

    y_pred_test = generate_preds_softmax(np.mean(preds, 0), (settings.ORIG_H, settings.ORIG_W))

    submission = create_submission(meta, y_pred_test)
    submission_filepath = 'ensemble_depths_res34_0123_tta2_1.csv'
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')

def generate_preds(outputs, target_size):
    preds = []

    for output in outputs:
        cropped = crop_image(output, target_size=target_size)
        pred = binarize(cropped, 0.5)
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
    #checkpoints = [
    #    r'G:\salt\models\152_new\best_0.pth', r'G:\salt\models\152_new\best_1.pth',
    #    r'G:\salt\models\152_new\best_2.pth'
    #]
    #checkpoints = [ LB841
    #    r'D:\data\salt\models\UNetResNetV4_34\best_0.pth', r'D:\data\salt\models\UNetResNetV4_34\best_1.pth',
    #    r'D:\data\salt\models\UNetResNetV4_34\best_2.pth'#, r'D:\data\salt\models\UNetResNetV4_34\best_3.pth'
    #]
    
    # LB856
    checkpoints = [
        r'D:\data\salt\models\depths\UNetResNetV4_34\best_0.pth', 
        r'D:\data\salt\models\depths\UNetResNetV4_34\best_1.pth',
        r'D:\data\salt\models\depths\UNetResNetV4_34\best_2.pth',
        r'D:\data\salt\models\depths\UNetResNetV4_34\best_3.pth'
    ]
    #predict()
    ensemble(checkpoints)
