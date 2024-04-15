"""
    创建数据集
"""

import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import os
from PIL import Image,ImageDraw
import math
from utils import *
from config import *

LABEL_FILE_PATH = "data/data.txt"
IMG_BASE_DIR = "data/images"

tf = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

def one_hot(cls_num, v):
    b = np.zeros(cls_num)
    b[v] = 1.
    return b

class YoloDataSet(Dataset):
    def __init__(self):
        f= open('data.txt','r')
        self.dataset = f.readlines() #读全部数据
       
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        #print(data)
        temp_data = data.split()
        _boxes = np.array([float(x) for x in temp_data[1:]])
        boxes =np.split(_boxes,len(_boxes)//5) # 每隔5份就做一个切割，是一组数据
        # print(boxes)
        img = make_416_image(os.path.join(IMG_BASE_DIR,temp_data[0]))
        img=img.resize((416,416))
        img_data = tf(img)
        # print(img_data.shape)

        label = {}
        for feature_size,anchor in anchors.items():
            label[feature_size] = np.zeros((feature_size,feature_size,3,5+CLASS_NUM))
    

if __name__ == '__main__':
    data = YoloDataSet()
    print(data[0])
    
    # print("============")
    # print(data[0][0][...,8])
    # print("============")
    # print(data[0][2][...,0])

