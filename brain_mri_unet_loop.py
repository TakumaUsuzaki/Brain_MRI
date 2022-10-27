import os
import gc
import cv2
import time
import tqdm
import random
import collections
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm as tq
from sklearn import preprocessing

from hausdorff import hausdorff_distance

import torch
import torchvision
import torch.nn as nn
import torch_optimizer as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset as BaseDataset

#import slack_message
import json
"""
GD-enhancing tumor (ET — label 4)
peritumoral edema (ED — label 2)
necrotic and non-enhancing tumor core (NCR/NET — label 1)
classes = ['gd_enhancing', 'edema', 'necrotic_area']
"""

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device:', device)


class UNet(nn.Module):

    def __init__(self, n_class, n_channel):
        super().__init__()
        self.n_class = n_class
        self.n_chennel = n_channel

        # contracting path 1
        self.conv1_1 = nn.Conv2d(self.n_chennel, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)

        # contracting path 2
        self.pool2 = nn.MaxPool2d(2, stride=2) # 1/2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)

        # contracting path 3
        self.pool3 = nn.MaxPool2d(2, stride=2) # 1/4
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)

        # contracting path 4
        self.pool4 = nn.MaxPool2d(2, stride=2) # 1/8
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)

        # contracting path 5
        self.pool5 = nn.MaxPool2d(2, stride=2) # 1/16
        self.conv5_1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)

        # expansive path 1
        self.dconv1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6_1 = nn.Conv2d(1024, 512, 3, padding=1)
        self.relu6_1 = nn.ReLU(inplace=True)
        self.conv6_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu6_2 = nn.ReLU(inplace=True)

        # expansive path 2
        self.dconv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7_1 = nn.Conv2d(512, 256, 3, padding=1)
        self.relu7_1 = nn.ReLU(inplace=True)
        self.conv7_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu7_2 = nn.ReLU(inplace=True)

        # expansive path 3
        self.dconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8_1 = nn.Conv2d(256, 128, 3, padding=1)
        self.relu8_1 = nn.ReLU(inplace=True)
        self.conv8_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu8_2 = nn.ReLU(inplace=True)

        # expansive path 4
        self.dconv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9_1 = nn.Conv2d(128, 64, 3, padding=1)
        self.relu9_1 = nn.ReLU(inplace=True)
        self.conv9_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu9_2 = nn.ReLU(inplace=True)
        self.conv9_3 = nn.Conv2d(64, self.n_class, 1)

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        output1 = self.relu1_2(self.conv1_2(h))

        h = self.pool2(output1)
        h = self.relu2_1(self.conv2_1(h))
        output2 = self.relu2_2(self.conv2_2(h))

        h = self.pool3(output2)
        h = self.relu3_1(self.conv3_1(h))
        output3 = self.relu3_2(self.conv3_2(h))

        h = self.pool4(output3)
        h = self.relu4_1(self.conv4_1(h))
        output4 = self.relu4_2(self.conv4_2(h))

        h = self.pool5(output4)
        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))

        upsample1 = self.dconv1(h)
        h = torch.cat((output4, upsample1), dim=1)
        h = self.relu6_1(self.conv6_1(h))
        h = self.relu6_2(self.conv6_2(h))

        upsample2 = self.dconv2(h)
        h = torch.cat((output3, upsample2), dim=1)
        h = self.relu7_1(self.conv7_1(h))
        h = self.relu7_2(self.conv7_2(h))

        upsample3 = self.dconv3(h)
        h = torch.cat((output2, upsample3), dim=1)
        h = self.relu8_1(self.conv8_1(h))
        h = self.relu8_2(self.conv8_2(h))

        upsample4 = self.dconv4(h)
        h = torch.cat((output1, upsample4), dim=1)
        h = self.relu9_1(self.conv9_1(h))
        h = self.relu9_2(self.conv9_2(h))
        h = self.conv9_3(h)

        return h


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def f_score(pr, gt, eps=1e-7, threshold=None):
    act = nn.Sigmoid()
    pr = act(pr)
    score = 0.0

    if threshold is not None:
        pr = (pr > threshold).float()

    for i in range(pr.shape[0]):
        tp = torch.sum(gt[i, 0, :, :] * pr[i, 0, :, :])
        fp = torch.sum(pr[i, 0, :, :]) - tp
        fn = torch.sum(gt[i, 0, :, :]) - tp

        score += (2*tp + eps)/(2*tp + fn + fp + eps)

    return score


def iou(pr, gt, eps=1e-7, threshold=None):
    act = nn.Sigmoid()
    pr = act(pr)

    score = 0.0

    if threshold is not None:
        pr = (pr > threshold).float()

    for i in range(pr.shape[0]):
        #plt.imshow(pr[i, 0, :, :]*255)
        #plt.show()
        #plt.imshow(gt[i, 0, :, :]*255)
        #plt.show()
        tp = torch.sum(gt[i, 0, :, :] * pr[i, 0, :, :])
        fp = torch.sum(pr[i, 0, :, :]) - tp
        fn = torch.sum(gt[i, 0, :, :]) - tp

        score += (tp + eps)/(tp + fn + fp + eps)

    return score


def hd95(pr, gt, eps=1e-7, threshold=None):
    act = nn.Sigmoid()
    pr = act(pr)
    score = 0.0
    pr = pr.to('cpu').detach().numpy()
    gt = gt.to('cpu').detach().numpy()

    if threshold is not None:
        pr = (pr > threshold).float()

    _, _, m, n = pr.shape

    for i in range(pr.shape[0]):
        v = np.array(np.where(pr[i, 0, :, :] > 0))
        v = np.array([[v[0, i], v[1, i]] for i in range(v.shape[1])])
        u = np.array(np.where(gt[i, 0, :, :] > 0))
        u = np.array([[u[0, i], u[1, i]] for i in range(u.shape[1])])

        score += hausdorff_distance(u, v, distance='euclidean')

    return score


class BCEDiceLoss(nn.Module):

    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class Dataset(BaseDataset):

    CLASSES = ['edema']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
    ):

        self.ids = os.listdir(images_dir)
        self.ids_mask = os.listdir(masks_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids_mask]

        # convert str names to class values on masks
        self.class_values = [2**self.CLASSES.index(cls.lower()) for cls in classes]


    def __getitem__(self, i):

        # read data
        """
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        """
        image = np.load(self.images_fps[i])
        #image = image+(image**2/np.max(image**2))*255

        image_e = image-(image**5/np.max(image**5))*255
        image_e = (image_e**5/np.max(image_e**5))*255
        #image = image + image_e

        #image_exp2 = (image-(image**2np.max(image**20))*255)**10
        #image_exp2 = image_exp2/np.max(image_exp2)
        image = image/np.max(image)
        #image = image + image_exp1
        image_e = image_e/np.max(image_e)
        image = np.stack([image_e], axis=-1).astype('float')
        #for j in range(3):
        #    image[:,:,j] = image[:,:,j]/np.max(image[:,:,j])
        #image = np.concatenate([image, image], axis=2).astype('float')
        image = image.transpose(2, 0, 1).astype('float32')
        image = torch.from_numpy(image).clone()

        mask = np.load(self.masks_fps[i])
        mask = np.where(mask > 0, 1., 0.)

        #masks = [(mask == v) for v in self.class_values]
        mask = np.stack([mask], axis=-1).astype('float')
        #print(mask.shape)
        #print(np.max(mask), np.min(mask))

        #plt.imshow(mask)
        #plt.show()
        mask = mask.transpose(2, 0, 1).astype('float32')
        mask = torch.from_numpy(mask).clone()

        return image, mask

    def __len__(self):
        return len(self.ids)



classes = ['edema']

DATA_DIR = 'C:/Users/kuma/Dropbox/brain_mri/train_val_test/'

x_train_dir = os.path.join(DATA_DIR, 'train_t1Gd')
y_train_dir = os.path.join(DATA_DIR, 'train_ann')

x_valid_dir = os.path.join(DATA_DIR, 'val_t1Gd')
y_valid_dir = os.path.join(DATA_DIR, 'val_ann')

x_test_dir = os.path.join(DATA_DIR, 'test_t1Gd')
y_test_dir = os.path.join(DATA_DIR, 'test_ann')

train_dataset = Dataset(
    x_train_dir,
    y_train_dir,
    classes=classes,
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
    )


valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=classes,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
    )

test_dataset = Dataset(
    x_test_dir,
    y_test_dir,
    classes=classes,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
    )


def deep_learning_loop(epoch_n, ith):

    model = UNet(1, 1).to(device)
    # model_path = 'C:/Users/kuma/Dropbox/brain_mri/Script/model.pth'
    # model.load_state_dict(torch.load(model_path))

    criterion = BCEDiceLoss()
    optimizer = optim.RAdam(model.parameters(), lr=0.001)
    current_lr = [param_group['lr'] for param_group in optimizer.param_groups][0]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2, cooldown=2)

    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []
    dice_score_val_list = []
    IoU_val_list = []
    HD95_val_list = []
    dice_score_test_list = []
    IoU_test_list = []
    HD95_test_list = []
    lr_rate_list = []
    valid_loss_min = np.Inf

    for epoch in range(1, epoch_n+1):

        train_loss = 0.0
        valid_loss = 0.0
        test_loss = 0.0
        dice_score_val = 0.0
        IoU_score_val = 0.0
        HD95_score_val = 0.0
        dice_score_test = 0.0
        IoU_score_test = 0.0
        HD95_score_test = 0.0

        ###################
        # train the model #
        ###################
        model.train(True)

        bar = tq(train_loader, postfix={"train_loss":0.0})
        for data, target in bar:

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

            bar.set_postfix(ordered_dict={"train_loss":loss.item()})

        ######################
        # validate the model #
        ######################
        model.eval()
        del data, target
        with torch.no_grad():
            bar = tq(valid_loader, postfix={"valid_loss":0.0})
            for data, target in bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()*data.size(0)
                dice_score_val += f_score(output.cpu(), target.cpu())
                IoU_score_val += iou(output.cpu(), target.cpu())
                if epoch == epoch_n:
                    HD95_score_val += hd95(output.cpu(), target.cpu())
                bar.set_postfix(ordered_dict={"valid_loss": loss.item()})


        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        dice_score_val = dice_score_val.item()/len(valid_loader.dataset)
        IoU_score_val = IoU_score_val.item()/len(valid_loader.dataset)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        dice_score_val_list.append(dice_score_val)
        IoU_val_list.append(IoU_score_val)
        lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])

        if epoch == epoch_n:
            HD95_score_val = HD95_score_val/len(valid_loader.dataset)
            HD95_val_list.append(HD95_score_val)


        # print training/validation statistics
        print('{} th loop  Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f} Dice Score: {:.6f} IoU: {:.6f}'.format(ith, epoch, train_loss, valid_loss, dice_score_val, IoU_score_val))

        #if epoch % 10 == 0:
        #    slack_message.reoportor('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f} Dice Score: {:.6f} IoU: {:.6f}'.format(
        #        epoch, train_loss, valid_loss, dice_score_val, IoU_score_val))


        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving mode...'.format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), 'model.pth')
            valid_loss_min = valid_loss

        ######################
        # test the model #
        ######################

        del data, target
        with torch.no_grad():
            bar = tq(test_loader, postfix={"test_loss": 0.0})
            for data, target in bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()*data.size(0)
                dice_score_test += f_score(output.cpu(), target.cpu())
                IoU_score_test += iou(output.cpu(), target.cpu())
                if epoch == epoch_n:
                    HD95_score_test += hd95(output.cpu(), target.cpu())
                bar.set_postfix(ordered_dict={"test_loss": loss.item()})

        # calculate average losses
        test_loss = test_loss/len(valid_loader.dataset)
        dice_score_test = dice_score_test.item()/len(valid_loader.dataset)
        IoU_score_test = IoU_score_test.item()/len(valid_loader.dataset)
        test_loss_list.append(test_loss)
        dice_score_test_list.append(dice_score_test)
        IoU_test_list.append(IoU_score_test)
        if epoch == epoch_n:
            HD95_test_list.append(HD95_score_test)
            HD95_score_test = HD95_score_test/len(valid_loader.dataset)

        # print training/validation statistics
        print('{} th loop  Epoch: {}  Test Loss: {:.6f}  Dice Score: {:.6f} IoU: {:.6f}'.format(ith,
            epoch, test_loss, dice_score_test, IoU_score_test))

        #if epoch % 30 == 0:
        #    slack_message.reoportor('Epoch: {}  Test Loss: {:.6f} Dice Score: {:.6f} IoU: {:.6f}'.format(epoch, test_loss, dice_score_test, IoU_score_test))

        scheduler.step(valid_loss)

    result = {
        'train_loss_list': train_loss_list,
        'valid_loss_list': valid_loss_list,
        'test_loss_list': test_loss_list,
        'dice_score_val_list': dice_score_val_list,
        'IoU_val_list': IoU_val_list,
        '95HD_val_list': HD95_val_list,
        'dice_score_test_list': dice_score_test_list,
        'IoU_test_list': IoU_test_list,
        '95HD_test_list': HD95_test_list,
        'lr_rate_list': lr_rate_list
    }

    with open('C:/Users/kuma/Dropbox/brain_mri/Script/t1Gdexp/result_t1Gdexp_{}.json'.format(ith), 'w') as f:
        json.dump(result, f, indent=4)

    torch.cuda.empty_cache()

    return()


for i in range(20):
    deep_learning_loop(150, i+1)
