import os
import numpy as np
import cv2

def process_fold(fold):
    images = np.load('data/' + fold + '/images/' + fold + '/images.npy')
    types = np.load('data/' + fold + '/images/' + fold + '/types.npy')
    masks = np.load('data/' + fold + '/masks/' + fold + '/masks.npy')
    # images = np.zeros((100,32,32,3))
    # masks = np.zeros((100,32,32,6))

    if not os.path.exists('data/' + fold + '_images'):
        os.makedirs('data/' + fold + '_images')
    if not os.path.exists('data/' + fold + '_masks'):
        os.makedirs('data/' + fold + '_masks')

    for i in range(0, images.shape[0]):
        cv2.imwrite(os.path.join('data/' + fold + '_images', str(i).zfill(5) + '.png'),
                    images[i].astype(np.uint8))
        mask = masks[i, :, :, 0] + \
                masks[i, :, :, 1] + \
                masks[i, :, :, 2] + \
                masks[i, :, :, 3] + \
                masks[i, :, :, 4] + \
                masks[i, :, :, 5]

        cv2.imwrite(os.path.join('data/' + fold + '_masks', str(i).zfill(5) + '.png'),
                    mask.astype(np.uint8))

process_fold('fold_1')
process_fold('fold_2')
process_fold('fold_3')
