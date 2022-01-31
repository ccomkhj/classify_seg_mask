import cv2
import numpy as np
import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--mask', default='mask', help='the dir to binary mask')
    parser.add_argument('--output', default='output', help='the dir to output')
    args = parser.parse_args()
    return args

def group(mask, args, debug=False, save=True):
    img = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY)
    sketch = img.copy()
    num_groups, group_mask, bboxes, centers  = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    ''' 
    bboxes: left most x, top most y, horizontal size, vertical size, total area 
    '''
    MIN_AREA = 2000

    cropped_imgs = []

    for i, (bboxe, center) in enumerate(zip(bboxes, centers)):
        tx, ty, hori, verti, area = bboxe
        if area < MIN_AREA or i == 0:
            # skip too small or the whole image
            continue
        cx, cy = center

        cropped = img[ty:ty+verti, tx:tx+hori]

        if save:
            # TODO: check if there is only one object in the rect.
            file_name = os.path.basename(mask).split('.')
            cv2.imwrite(os.path.join(args.output, file_name[0] + f"_{str(i)}."+ file_name[1]), cropped*255)

        cropped_imgs.append(cropped)

        if debug:
            sketch = cv2.rectangle(sketch, pt1= (tx, ty), pt2= (tx+hori, ty+verti), color= 1, thickness= 2)
            f, axarr = plt.subplots(2)
            axarr[0].imshow(sketch)
            axarr[1].imshow(cropped)
            plt.show()
    return cropped_imgs
        
        


        


if __name__ == '__main__':
    args = parse_args()
    masks = glob.glob(os.path.join(args.mask, '*'))
    for mask in masks:
        cropped_imgs = group(mask, args)


        

        