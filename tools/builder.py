from torch.utils.data import Dataset

import cv2
import glob
import numpy as np
import random
import copy

from loguru import logger
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')




def flatten(t):
    return [item for sublist in t for item in sublist]

####################################################
#       Create Train, Valid and Test sets
####################################################
def set_files(train_data_path, test_data_path):
    # train_data_path = 'images/train' 
    # test_data_path = 'images/test'

    train_image_paths = [] #to store image paths in list
    classes = [] #to store class values

    #1.
    # get all the paths from train_data_path and append image paths and class to to respective lists
    # eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
    # eg. class -> 26.Pont_du_Gard
    for data_path in glob.glob(train_data_path + '/*'):
        classes.append(data_path.split('/')[-1]) 
        train_image_paths.append(glob.glob(data_path + '/*'))
        
    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    logger.info('train_image_path example: ', train_image_paths[0])
    logger.info(f'classes: {classes}')

    #2.
    # split train valid from train paths (80,20)
    train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):] 

    #3.
    # create the test_image_paths
    test_image_paths = []
    for data_path in glob.glob(test_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = list(flatten(test_image_paths))

    logger.info("Train size: {}, Valid size: {}, Test size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

    #######################################################
    #      Create dictionary for class indexes
    #######################################################

    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}

    return train_image_paths, valid_image_paths, test_image_paths, class_to_idx 



#######################################################
#               Define Dataset Class
#######################################################

class SymbolDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, image_type, transform=False):
        self.image_paths = image_paths
        self.transform = transform
        self.class_to_idx = class_to_idx 
        self.image_type = image_type
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        if self.image_type == 'RGB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        else:
            image = image

        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    
#######################################################
#                  Visualize Dataset
#         Images are plotted after augmentation
#######################################################

def visualize_augmentations(dataset, idx=0, samples=10, cols=5, random_img = False):
    
    dataset = copy.deepcopy(dataset)
    #we remove the normalize and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    
        
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1,len(train_image_paths))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(idx_to_class[lab])
    plt.tight_layout(pad=1)
    plt.show()    

# visualize_augmentations(train_dataset, np.random.randint(1,len(train_image_paths)), random_img = True)
