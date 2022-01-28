from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
from tools.builder import SymbolDataset, set_files
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import cv2
from loguru import logger


class Net(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_ch, out_channels = 16, kernel_size = 3, stride= 1) 
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, stride = 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(73728, 128)
        self.fc2 = nn.Linear(128, out_ch) 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--train_path', default='images/train', help='the dir to training images')
    parser.add_argument('--test_path', default='images/test', help='the dir to testing images')

    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
                        

    args = parser.parse_args()
    return args

def main(image_type):

    # Training settings
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    #######################################################
    #               Define Transforms
    #######################################################

    #To define an augmentation pipeline, you need to create an instance of the Compose class.
    #As an argument to the Compose class, you need to pass a list of augmentations you want to apply. 

    train_transforms = A.Compose([
            A.Rotate(limit= [-180,180], border_mode= cv2.BORDER_CONSTANT, value=0, mask_value= 1, p=1),  # mask value could be 255. It should be verified.,
            A.Resize (height=100, width=100, p=1),
            # A.SmallestMaxSize(max_size=350), # Rescale an image so that minimum side is equal to max_size, keeping the aspect ratio of the initial image.
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, p=0.5),
            # A.RandomCrop(height=256, width=256),
            # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            # A.RandomBrightnessContrast(p=0.5),
            # A.MultiplicativeNoise(multiplier=[0.5,2], per_channel=True, p=0.2),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            # A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            ToTensorV2(),
        ])

    test_transforms = A.Compose([
            # A.SmallestMaxSize(max_size=350),
            # A.CenterCrop(height=256, width=256),
            # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.Resize (height=100, width=100, p=1),
            ToTensorV2(),
        ])

    train_image_paths, valid_image_paths, test_image_paths, class_to_idx  = set_files(args.train_path, args.test_path)

    #######################################################
    #                  Create Dataset
    #######################################################

    train_dataset = SymbolDataset(train_image_paths, class_to_idx, image_type, train_transforms)
    valid_dataset = SymbolDataset(valid_image_paths, class_to_idx, image_type, test_transforms) #test transforms are applied
    test_dataset = SymbolDataset(test_image_paths, class_to_idx, image_type, test_transforms)

    logger.info('The shape of tensor for the first image in train dataset: ',train_dataset[0][0].shape)
    logger.info('The label for the first image in train dataset: ',train_dataset[0][1])

    #######################################################
    #                  Define Dataloaders
    #######################################################

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if image_type == 'RGB': in_ch = 3
    else: in_ch = 1
    num_class = len(class_to_idx)

    model = Net(in_ch, num_class).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "test.pt")


if __name__ == '__main__':

    main(image_type = 'RGB') # RGB or Gray