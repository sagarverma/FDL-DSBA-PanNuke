""" Submit code specific to the kaggle challenge"""

import os

import torch
from PIL import Image
import numpy as np

from predict import predict_img
from unet import UNet #Load your model class

def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def submit(net):
    """Used for Kaggle submission: predicts and encode all test images"""
    dir = 'data/test_images/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = len(list(os.listdir(dir)))
    with open('UNet_SUBMISSION.csv', 'w') as f:
        f.write('img,rle_mask\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))

            img = Image.open(dir + i)

            mask = predict_img(net, img, device)
            enc = rle_encode(mask)
            f.write('{},{}\n'.format(i, enc))


if __name__ == '__main__':
    net = UNet(3, 1).cuda() # Instansiate your model
    net.load_state_dict(torch.load('checkpoints/CP_epoch5.pth')) # Load your model trained weights
    submit(net)
