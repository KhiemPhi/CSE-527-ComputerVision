# import packages here
from __future__ import print_function, division
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import random 
import pandas as pd
from skimage import io, transform
import numpy as np
import cv2
import copy
import scipy.io
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from random import shuffle


##Import the VGG16 model
import torchvision.models as models
model = models.vgg16(pretrained = True)

##Modify the VGG16 net, only one fully connected layer
classifier = list(model.classifier.children())[:-4]
print(classifier)
model.classifier = nn.Sequential(*classifier)
print(model)

##Getting the keys
def getKeys(path):
    keys = []
    file_path = sorted(glob.glob(path))
    for path in file_path:
        image_path = sorted(glob.glob(path + '/*.jpg'))
        for i in image_path:
            key = i[52:60]
        #print(key)
        keys.append(key)
    return keys

def featureGenerator(image):
    cropped1 = image[16:240, 58:282]
    cropped2 = image[0:224, 0:224]
    cropped3  = image[32:256, 0:224]
    cropped4 = image[0:224, 116:340]
    cropped5 = image[32:256, 116:340]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    prep = transforms.Compose([ transforms.ToTensor(), normalize ])
    inputs = [prep(cropped1), prep(cropped2), prep(cropped3), prep(cropped4), prep(cropped5)]
    inputs = torch.stack(inputs)
    outputs = model(inputs)
    feature = outputs.mean(0)
    return feature

path = "UCF101_dimitris_course/UCF101_release/images_class1/*"
keys = getKeys(path)

print(len(keys))
print(keys)

#Getting images in batches#
batch = []
images = []
empty = []
file_path = sorted(glob.glob(path))
for path in file_path:
    #print("FIle Path:",path)
    image_path = sorted(glob.glob(path + '/*.jpg'))
    batch = copy.deepcopy(empty)
    for i in image_path:
        #print("image path:",i)
        batch.append(cv2.imread(i))
        #print(len(batch))
    images.append(batch)

print(len(images))
print(len(images[0]))

##Dictionary of images and corresponding keys
image_data = dict(zip(keys,images))

featSet1 = []
featSet2 = []
empty = []
features = {}
empty_dict = {}

##Generating featurs and saving them in a .mat file
for key in image_data:
    print(key)
    count = 0
    featSet1 = copy.deepcopy(empty)
    featSet2 = copy.deepcopy(empty)
    features = copy.deepcopy(empty_dict)
    for image in image_data[key]:
        #print("before:", count)
        feature = featureGenerator(image)
        #print(feature.shape)
        #featSet.append(feature)
        #print(len(featSet))
        if 13 > count >= 0:
            featSet1.append(feature)
            #print(len(featSet1))
            features['Features1'] = torch.stack(featSet1).detach().numpy()
            #scipy.io.savemat(key, features)
        elif 25 > count >= 13:
            featSet1 = copy.deepcopy(empty)
            featSet2.append(feature)
            #print(len(featSet2))
            features['Features2'] = torch.stack(featSet2).detach().numpy()
        scipy.io.savemat(key,features)
        count += 1
        #print("after:", count)

for name in glob.glob('feats_generated/*'):
    file = scipy.io.loadmat(name)
    #print(file)
    feat_1 = file['Features1']
    feat_2 = file['Features2']
    features = np.concatenate((feat_1, feat_2))
    #print(features.shape)
    file['Feature'] = features
    scipy.io.savemat(name, file)

for name in glob.glob('feats_generated/*'):
    file = scipy.io.loadmat(name)
    #print(file)
    print(file['Feature'].shape)