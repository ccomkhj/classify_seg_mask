import cv2
import numpy as np
import argparse
import glob
import os
from loguru import logger

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--mask', default='mask', help='the dir to binary mask')
    parser.add_argument('--output', default='output', help='the dir to output')
    args = parser.parse_args()
    return args

def group(img, mask, out_dir, debug=False, save=True):
    
    sketch = img.copy()
    num_groups, group_mask, bboxes, centers  = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    logger.info(f'The number of {num_groups} groups are detected.')
    ''' 
    bboxes: left most x, top most y, horizontal size, vertical size, total area 
    '''
    MIN_AREA = 1000

    cropped_masks = []
    rois = []

    for i, (bboxe, center) in enumerate(zip(bboxes, centers)):
        tx, ty, hori, verti, area = bboxe
        if area < MIN_AREA or i == 0:
            # skip too small or the whole image
            continue
        cx, cy = center
        roi = ty, ty+verti, tx, tx+hori

        cropped = mask[roi[0]:roi[1], roi[2]:roi[3]]

        if save and cv2.connectedComponents(cropped)[0] == 2: # if there is only one object in the cropped mask,
                file_name = os.path.basename(mask).split('.')
                cv2.imwrite(os.path.join(out_dir, file_name[0] + f"_{str(i)}."+ file_name[1]), cropped*255)
                logger.info(f'{file_name[0]}_{str(i)} is saved')

        cropped_masks.append(cropped)
        rois.append(roi)

        if debug:
            sketch = cv2.rectangle(sketch, pt1= (tx, ty), pt2= (tx+hori, ty+verti), color= 1, thickness= 2)
            f, axarr = plt.subplots(2)
            axarr[0].imshow(sketch)
            axarr[1].imshow(cropped)
            plt.show()
    return cropped_masks, rois
        
if __name__ == '__main__':
    args = parse_args()
    masks = glob.glob(os.path.join(args.mask, '*'))
    for mask in masks:
        logger.info('This image is being processed : ', mask)
        img = cv2.cvtColor(cv2.imread(mask), cv2.COLOR_BGR2GRAY)
        cropped_masks = group(img, mask, args.output)


        

        