import torch
from torch import nn 
from yolo_v3_net import Yolo_V3_Net
import os

class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_path= 'params/net597.pt'

        self.net = Yolo_V3_Net().to(device)
        if os.path.exists(self.weight_path):
            self.net.load_state_dict(torch.load(self.weight_path))
        
        self.net.eval() # 加载batch参数加载到预测过程中
        
    def forward(self, input, thresh, anchors,case):
        pass

    def get_index_and_bias(self,output,thresh):
        output = output.permute(0, 2, 3, 1)#
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        
        
