import os

from PIL import Image
import numpy as np

def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs



def creat_rles():
    """Used for Kaggle submission: predicts and encode all test images"""
    dir = 'data/test_masks/'
    N = len(list(os.listdir(dir)))
    with open('answer_key.csv', 'a') as f:
        f.write('img,rle_mask\n')
        for index, i in enumerate(os.listdir(dir)):
            print('{}/{}'.format(index, N))

            mask = Image.open(dir + i)
            mask = np.array(mask)

            enc = rle_encode(mask // 255)
            f.write('{},{}\n'.format(i, ' '.join(map(str, enc))))
