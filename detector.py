import torch
from torch import nn 
from yolo_v3_net import Yolo_V3_Net
import os
from PIL import Image,ImageDraw
from utils import *
from dataset import * 
from config import* 

class_num={
    0:'person',
    1:'horse',
    2:'bicycle',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.weight_path= 'params/net178.pt'
        self.net = Yolo_V3_Net().to(device)
        if os.path.exists(self.weight_path):
            self.net.load_state_dict(torch.load(self.weight_path))
        
        self.net.eval() # 加载batch参数加载到预测过程中
        
    def forward(self, input, thresh, anchors,case):
        output_13, output_26, output_52 = self.net(input)
        idxs_13, bias_13 = self.get_index_and_bias(output_13, thresh)
        boxes_13 = self.get_true_position(idxs_13, bias_13, 32, anchors[13],case)

        idxs_26, bias_26 = self.get_index_and_bias(output_26, thresh)
        boxes_26 = self.get_true_position(idxs_26, bias_26, 16, anchors[26],case)

        idxs_52, bias_52 = self.get_index_and_bias(output_52, thresh)
        boxes_52 = self.get_true_position(idxs_52, bias_52, 8, anchors[52],case)

        return torch.cat([boxes_13,boxes_26,boxes_52],dim=0)

    def get_index_and_bias(self,output,thresh):
        output = output.permute(0, 2, 3, 1)#
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1) # N H W 3 8
        
        mask = output[...,0]>thresh #N H W 3 #最后一个维度的第0个元素，代表置信度
        index = mask.nonzero() # 返回为true的坐标索引
        bias = output[mask]

        return index,bias
    
    def get_true_position(self,index,bias,t,anchors,case): # 多加一个case参数，处理416*416到原图的缩放
        anchors = torch.tensor(anchors)
        a = index[:,3] # 拿到了每个框
        cy = (index[:,1].float()+bias[:,2].float())*t/case # 需要区分是13*13特征图上的，还是26*26,52*52上的，按照t比例还原
        cx = (index[:,2].float()+bias[:,1].float())*t/case # 需要区分是13*13特征图上的，还是26*26,52*52上的，按照t比例还原

        w = anchors[a,0]*torch.exp(bias[:,3])/case
        h = anchors[a,1]*torch.exp(bias[:,4])/case

        p = bias[:,0]
        cls_p = bias[:,5:]
        cls_index = torch.argmax(cls_p,dim=1)

        return torch.stack([torch.sigmoid(p),cx,cy,w,h,cls_index],dim=1)


if __name__ == '__main__':
    detector = Detector()
    img = Image.open('/home/liuchang/codetest/PythonCode/YOLOv3/data/images')
    _img = make_416_image('/home/liuchang/codetest/PythonCode/YOLOv3/data/images')
    temp= max(_img.size())
    case = 416/temp

    _img = _img.resize((416,416))
    
    _img = tf(_img).to(device)

    _img = torch.unsqueeze(_img,dim=0) # 数据升一个维度
    results = detector(_img,0.3,anchors,case)

    draw = ImageDraw.Draw(img)

    for rst in results:
        x1,y1,x2,y2 = rst[1]-0.5*rst[3],rst[2]-0.5*rst[4],rst[1]-0.5*rst[3],rst[2]-0.5*rst[4]
        print(x1,y1,x2,y2)
        print('class',class_num[int(rst[5])])

        draw.text((x1,y1),str(class_num[int(rst[5].item())])+str(rst[0].item())[:4]) #tensor类型数据要取值的话，要加一个item()
        draw.rectangle((x1,y1,x2,y2),width=1,outline='red')
    
    img.show()