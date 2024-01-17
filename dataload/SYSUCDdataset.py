from PIL import Image
import os
import glob
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data
import random
import torchvision.transforms.functional as tf

root_path = r'C:\PycharmProjects\pytorch-from book\codes-paper2\datasets\SYSU-CD'

def read_images(root):
    imgs_At = glob.glob(root + '/train/A/*.png')
    imgs_Bt = glob.glob(root + '/train/B/*.png')
    labelst = glob.glob(root + '/train/OUT/*.png')
    imgs_Att = glob.glob(root + '/test/A/*.png')
    imgs_Btt = glob.glob(root + '/test/B/*.png')
    labelstt = glob.glob(root + '/test/OUT/*.png')
    return imgs_At, imgs_Bt, labelst, imgs_Att, imgs_Btt, labelstt



def imagestransforms(before, after, change):
    if random.random() > 0.7:
        before = tf.hflip(before)
        after = tf.hflip(after)
        change = tf.hflip(change)
    if random.random() < 0.3:
        before = tf.vflip(before)
        after = tf.vflip(after)
        change = tf.vflip(change)
    angle = transforms.RandomRotation.get_params([-180, 180])
    before = before.rotate(angle)
    after = after.rotate(angle)
    change = change.rotate(angle)

    before = tf.to_tensor(before) #
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    return before, after, change

def images_transforms(before, after, change):
    before, after, change = imagestransforms(before, after, change)
    return before, after, change

def images_transforms_(before, after, change):
    before = tf.to_tensor(before)
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    return before, after, change

class SYSUCDDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        # self.before_list, self.after_list, self.change_list = 
        self.train_dataset_before, self.train_dataset_after, self.train_dataset_change, self.test_dataset_before, self.test_dataset_after, self.test_dataset_change = read_images(root_path)
        if mode == 'train':

            print('训练集加载了' + str(len(self.train_dataset_before)) + '张图片')
        elif mode == 'test':
            print('测试集加载了' + str(len(self.test_dataset_before)) + '张图片')

    def __getitem__(self, item):
        if self.mode == 'train':

            before = Image.open(self.train_dataset_before[item]).convert('RGB') #打开并转换成rgb图像
            after = Image.open(self.train_dataset_after[item]).convert('RGB')
            change = Image.open(self.train_dataset_change[item]).convert('L') #打开并转换为灰度图像
            before, after, change = images_transforms(before, after, change)
            # print("代码执行到这里了")

            # print('before shape:', before.shape)
            # print('after shape:', after.shape)
            # print('change shape:', change.shape)

            return before, after, change
        elif self.mode == 'test':
            before = Image.open(self.test_dataset_before[item]).convert('RGB')
            after = Image.open(self.test_dataset_after[item]).convert('RGB')
            change = Image.open(self.test_dataset_change[item]).convert('L')
            before, after, change = images_transforms_(before, after, change)
            # print('before shape:', before.shape)
            # print('after shape:', after.shape)
            # print('change shape:', change.shape)

            return before, after, change


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset_before)
        elif self.mode == 'test':
            return len(self.test_dataset_before)

if __name__ == '__main__':
    train_data = SYSUCDDataset(mode='train')
    for i in range(1):
        before, after, change = train_data[i]

    test_data = SYSUCDDataset(mode='test')
    for i in range(1):
        before, after, change = test_data[i]


