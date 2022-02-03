from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import argparse
import torch
import numpy as np
from tools.groupping import group
from tools.classification import predict, Net
import cv2
import matplotlib.pyplot as plt
from loguru import logger

plant_targets = ['basil', 'kopf salad']
symbol_targets = ['Left', 'Left_Right', 'Right', 'Straight', 'Straight_Left', 'Straight_Right', 'Unknown']

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Convert VIA dataset into Customdataset for mmsegmentation")
    parser.add_argument("--config", default =  r'/home/hexaburbach/codes/mmsegmentation/work_dirs/fcn_unet_s5-d16_128x128_40k_LeafDataset_T14/fcn_unet_s5-d16_128x128_40k_LeafDataset_T14.py',
                        help="config file.")
    parser.add_argument("--seg_weight", default = r'/home/hexaburbach/codes/mmsegmentation/work_dirs/fcn_unet_s5-d16_128x128_40k_LeafDataset_T14/iter_40000.pth',
                        help="pretrained segmentation weight file in pth form.")
    parser.add_argument("--cls_weight", default = r'checkpoints/hexa_v0.pt',
                        help="pretrained classification weight file in pt form.")
    parser.add_argument("--input", default = r'20220202_095522.jpg',
                        help="Specify the input image location."),
    parser.add_argument("--output", default='output',
                        help="Specify the folder location to save segmentation.")

    args = parser.parse_args()
    return args

def main(device='cuda:0'):
    
    seg_model = init_segmentor(args.config, args.seg_weight, device=device)
    cls_model = Net(in_ch=3, out_ch = len(plant_targets))
    cls_model.load_state_dict(torch.load(args.cls_weight))
    cls_model.eval()

    img = mmcv.imread(args.input) 
    img = mmcv.imrescale(img, (0.4)) # only if phone camera is used.
    logger.debug("Proceed Image Segmentation...")
    mask = inference_segmentor(seg_model, img)[0]
    logger.debug("Image Segmentation Done.")

    cropped_masks, rois = group(img, mask.astype(np.uint8), args.output, debug=False, save=False)
    logger.debug("Each mask is separated. Each will be classified.")

    for cropped_mask, roi in zip(cropped_masks, rois):

        cropped_img = img[roi[0]:roi[1], roi[2]:roi[3]]
        species = predict(cls_model, cropped_img, plant_targets)
        regression_ratio = 1 / 100
        mass = cv2.countNonZero(cropped_mask) * regression_ratio 

        # Draw contour of detected symbol
        img = cv2.rectangle(img, pt1= (roi[2], roi[0]), pt2=(roi[3], roi[1]), color=(255,0,0), thickness=3 )
        # Put name above it
        img = cv2.putText(img, f'{species}, {int(mass)}g', (roi[2], roi[0]+4), cv2.FONT_HERSHEY_PLAIN, fontScale=2 ,color= (0, 255, 0), thickness=2)
        # plt.imshow(img)
        # plt.show()

    color_mask = np.zeros_like(img)
    color_mask[mask==1] = [255, 0, 255]
    return cv2.addWeighted(img, 0.7, color_mask, 0.3 , 0)


if __name__ == '__main__':
    args = parse_args()
    result = main()
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()
    logger.info("Successfully Done.")
