import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize
import cv2

from utils import get_crop_pad_sequence
import pdb

def resize_image(image, target_size):
    """Resize image to target size

    Args:
        image (numpy.ndarray): Image of shape (H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Resized image of shape (H x W).

    """
    #n_channels = image.shape[0]
    #resized_image = resize(image, (target_size[0], target_size[1]), mode='constant', anti_aliasing=True)
    resized_image = cv2.resize(image, target_size)
    return resized_image


def crop_image(image, target_size):
    """Crop image to target size. Image cropped symmetrically.

    Args:
        image (numpy.ndarray): Image of shape (C x H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Cropped image of shape (C x H x W).

    """
    top_crop, right_crop, bottom_crop, left_crop = get_crop_pad_sequence(image.shape[1] - target_size[0],
                                                                         image.shape[2] - target_size[1])
    cropped_image = image[:, top_crop:image.shape[1] - bottom_crop, left_crop:image.shape[2] - right_crop]
    return cropped_image

def crop_image_softmax(image, target_size):
    """Crop image to target size. Image cropped symmetrically.

    Args:
        image (numpy.ndarray): Image of shape (H x W).
        target_size (tuple): Target size (H, W).

    Returns:
        numpy.ndarray: Cropped image of shape (H x W).

    """
    top_crop, right_crop, bottom_crop, left_crop = get_crop_pad_sequence(image.shape[0] - target_size[0],
                                                                         image.shape[1] - target_size[1])
    #pdb.set_trace()
    cropped_image = image[top_crop:image.shape[0] - bottom_crop, left_crop:image.shape[1] - right_crop]
    return cropped_image

def binarize(image, threshold):
    image_binarized = (image > threshold).astype(np.uint8)
    return image_binarized
