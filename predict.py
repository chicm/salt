import os
import glob
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import settings
from loader import get_test_loader
from unet_models import UNetResNet
from unet_new import UNetResNetV4, UNetResNetV5
from postprocessing import crop_image, binarize, crop_image_softmax
from metrics import intersection_over_union, intersection_over_union_thresholds
from utils import create_submission

batch_size = 32

def do_tta_predict(model, ckp_path, tta_num=4):
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
    
    parent_dir = ckp_path+'_out'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    np_file = os.path.join(parent_dir, 'pred.npy')

    model_pred_result = np.mean(preds, 0)
    np.save(np_file, model_pred_result)

    return model_pred_result, meta

def predict():
    model = UNetResNet(152, pretrained=True, is_deconv=True)
    CKP = os.path.join(settings.MODEL_DIR, '152_new', 'best_0.pth')
    print('loading...', CKP)
    model.load_state_dict(torch.load(CKP))
    model = model.cuda()

    pred, meta = do_tta_predict(model, CKP)
    print(pred.shape)
    y_pred_test = generate_preds_softmax(pred, (settings.ORIG_H, settings.ORIG_W))

    submission = create_submission(meta, y_pred_test)
    submission_filepath = 'sub_tta_1.csv'
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')


def ensemble(checkpoints):
    model = UNetResNetV5(50)

    preds = []
    meta = None
    for checkpoint in checkpoints:
        model.load_state_dict(torch.load(checkpoint))
        model = model.cuda()
        print('predicting...', checkpoint)

        pred, meta = do_tta_predict(model, checkpoint, tta_num=4)
        preds.append(pred)

    y_pred_test = generate_preds_softmax(np.mean(preds, 0), (settings.ORIG_H, settings.ORIG_W))

    submission = create_submission(meta, y_pred_test)
    submission_filepath = 'ensemble_depths_res50_0123_8models.csv'
    submission.to_csv(submission_filepath, index=None, encoding='utf-8')

def ensemble_np(np_files):
    preds = []
    for np_file in np_files:
        pred = np.load(np_file)
        print(np_file, pred.shape)
        preds.append(pred)

    y_pred_test = generate_preds_softmax(np.mean(preds, 0), (settings.ORIG_H, settings.ORIG_W))

    meta = get_test_loader(batch_size, index=0, dev_mode=False).meta

    submission = create_submission(meta, y_pred_test)
    submission_filepath = 'ensemble_depths_res50_34_8model_tta2_1.csv'
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
    
    # LB861
    #checkpoints = [
    #    r'D:\data\salt\models\depths\UNetResNetV4_34\best_0.pth', 
    #    r'D:\data\salt\models\depths\UNetResNetV4_34\best_1.pth',
    #    r'D:\data\salt\models\depths\UNetResNetV4_34\best_2.pth',
    #    r'D:\data\salt\models\depths\UNetResNetV4_34\best_3.pth'
    #]

    #checkpoints= glob.glob(r'D:\data\salt\models\depths\UNetResNetV5_50\best*')
    #print(checkpoints)
    #ensemble(checkpoints)

    np_files1 = glob.glob(r'D:\data\salt\models\depths\UNetResNetV5_50\*pth_out\*.npy')
    np_files2 = glob.glob(r'D:\data\salt\models\depths\UNetResNetV4_34\*pth_out\*.npy')
    np_files = np_files1+np_files2
    print(np_files)
    ensemble_np(np_files)
