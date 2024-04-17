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
        f= open('./data.txt','r') #PythonCode/YOLOv3/data.txt
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
        w,h = img.size
        case = 416/w

        img=img.resize((DATA_WIDTH,DATA_HEIGHT))
        img_data = tf(img)
        # print(img_data.shape)

        # 制作数据集，生成先验框数据
        labels = {} 
        for feature_size,_anchor in anchors.items():
            labels[feature_size] = np.zeros((feature_size,feature_size,3,5+CLASS_NUM))
            # print(labels[feature_size].shape) #(13, 13, 3, 8)
            for box in boxes:
                cls,cx,cy,w,h = box
                cx,cy,w,h = cx*case,cy*case,w*case,h*case
                # print(cls,cx,cy,w,h)
                _x,x_index = math.modf(cx*feature_size/DATA_WIDTH)
                _y,y_index = math.modf(cy*feature_size/DATA_HEIGHT)
                
                for i,anchor in enumerate(_anchor):
                    area = w*h
                    iou = min(area,ANCHORS_AREA[feature_size][i])/max(area,ANCHORS_AREA[feature_size][i])
                    p_w,p_h = w/anchor[0],h/anchor[1]
                    index+=1
                    if labels[feature_size][int(x_index), int(y_index), i][0]<iou:
                        labels[feature_size][int(x_index), int(y_index), i] = np.array([iou, _x, _y, np.log(p_w), np.log(p_h), *one_hot(CLASS_NUM, int(cls))])
                
        return labels[13], labels[26], labels[52], img_data

if __name__ == '__main__':
    data = YoloDataSet()
    print(data[0][3].shape)
    print(data[0][2].shape)
    print(data[0][1].shape)
    
    # print("============")
    # print(data[0][0][...,8])
    # print("============")
    # print(data[0][2][...,0])

