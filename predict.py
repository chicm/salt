import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import settings
from loader import get_test_loader
from unet_models import UNetResNet
from postprocessing import crop_image, binarize, crop_image_softmax
from metrics import intersection_over_union, intersection_over_union_thresholds
from utils import create_submission

batch_size = 128
#CKP = 'models/152/best_814_elu.pth'
CKP = os.path.join(settings.MODEL_DIR, '152_new', 'best_0.pth')

def predict():
    model = UNetResNet(152, pretrained=True, is_deconv=True)
    print('loading...', CKP)
    model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    test_loader = get_test_loader(batch_size)

    model.eval()
    print('predicting...')
    outputs = []
    with torch.no_grad():
        for i, img in enumerate(test_loader):
            img = img.cuda()
            output, _ = model(img)
            output = torch.sigmoid(output)

            for o in output.cpu():
                outputs.append(o.squeeze().numpy())
            print(f'{batch_size*(i+1)} / {test_loader.num}', end='\r')

    y_pred_test = generate_preds_softmax(outputs, (settings.ORIG_H, settings.ORIG_W))

    submission = create_submission(test_loader.meta, y_pred_test)
    submission_filepath = 'sub_new2.csv'
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')

def ensemble(checkpoints):
    model = UNetResNet(152, pretrained=True, is_deconv=True)

    preds = []
    for checkpoint in checkpoints:
        model.load_state_dict(torch.load(checkpoint))
        model = model.cuda()

        test_loader = get_test_loader(batch_size)

        model.eval()
        print('predicting {}...'.format(checkpoint))
        outputs = []
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                img = img.cuda()
                output, _ = model(img)
                output = torch.sigmoid(output)

                for o in output.cpu():
                    outputs.append(o.squeeze().numpy())
                print(f'{batch_size*(i+1)} / {test_loader.num}', end='\r')
        preds.append(np.array(outputs).astype(np.float16))
        #print(preds[0].dtype)
    del model
    tmp = np.mean(preds, 0)
    tmp2 = []
    for i in range(len(tmp)):
        tmp2.append(tmp[i])

    y_pred_test = generate_preds_softmax(tmp2, (settings.ORIG_H, settings.ORIG_W))

    submission = create_submission(test_loader.meta, y_pred_test)
    submission_filepath = 'ensemble_152_new2.csv'
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
    '''
    checkpoints = [
        r'G:\salt\models\152\best_0.pth', r'G:\salt\models\152\best_1.pth',
        r'G:\salt\models\152\best_2.pth', r'G:\salt\models\152\best_3.pth', 
        r'G:\salt\models\152\best_4.pth', r'G:\salt\models\152\best_5.pth',
        r'G:\salt\models\152\best_6.pth', r'G:\salt\models\152\best_7.pth',
        r'G:\salt\models\152\best_8.pth', r'G:\salt\models\152\best_9.pth',
    ]
    
    checkpoints = [
        r'G:\salt\models\152\ensemble_822\best_0.pth', r'G:\salt\models\152\ensemble_822\best_1.pth',
        r'G:\salt\models\152\ensemble_822\best_2.pth', r'G:\salt\models\152\ensemble_822\best_3.pth',
        r'G:\salt\models\152\ensemble_822\best_4.pth'
    ]
    '''
    checkpoints = [
        r'G:\salt\models\152_new\best_0.pth', r'G:\salt\models\152_new\best_1.pth', r'G:\salt\models\152_new\best_2.pth'
    ]
    #predict()
    ensemble(checkpoints)
