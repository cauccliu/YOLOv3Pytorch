from torch import nn,optim
import torch
from dataset import *
from yolo_v3_net import *
from torch.utils.data import DataLoader # 数据加载器
from torch.utils.tensorboard import SummaryWriter # 训练可视化

def loss_fn(output, target,c):
    output = output.permute(0, 2, 3, 1)# 换轴 #N,45,13,13==>N,13,13,45
    output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)#N,13,13,3,15

    mask_obj = target[..., 0] > 0#N,13,13,3 # 正样本
    mask_noobj = target[..., 0] == 0 # 负样本

    loss_p_fun=nn.BCELoss() # 二分类损失
    loss_p=loss_p_fun(torch.sigmoid(output[...,0]),target[...,0]) # 正负样本都需要
    
    loss_box_fun=nn.MSELoss() # 回归损失
    loss_box=loss_box_fun(output[mask_obj][...,1:5],target[mask_obj][...,1:5])
    
    loss_segment_fun=nn.CrossEntropyLoss() # 多分类损失
    loss_segment = loss_segment_fun(output[mask_obj][...,5:],torch.argmax(target[mask_obj][...,5:],dim=1, keepdim= True).squeeze(dim=1))

    loss = c * loss_p + (1-c)*0.5*loss_box+ (1-c)*0.5*loss_segment
    return loss

if __name__ =='__main__':
    summary_writer = SummaryWriter('logs') # 可视化数据放在logs文件夹下

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 是否在cuda上训练
    dataset = YoloDataSet()
    data_Loader = DataLoader(dataset,batch_size=2,shuffle=True)

    weight_path= 'params/net597.pt' # 权重文件
    net = Yolo_V3_Net().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))

    opt = optim.Adam(net.parameters())

    epoch = 0
    while True:
        index = 0
        for target_13, target_26, target_52, img_data in data_Loader:
            # 数据放到GPU上
            target_13, target_26, target_52, img_data = target_13.to(device), target_26.to(device), target_52.to(device), img_data.to(device)
            # print(target_13.shape) # torch.Size([2, 13, 13, 3, 8])

            output_13, output_26, output_52 = net(img_data)
            loss_13 = loss_fn(output_13.float(), target_13.float(), 0.7)
            loss_26 = loss_fn(output_26.float(), target_26.float(), 0.7)
            loss_52 = loss_fn(output_52.float(), target_52.float(), 0.7)

            loss = loss_13 + loss_26 + loss_52
            opt.zero_grad()
            loss.backward()
            opt.step() # 梯度更新三件套

            print(epoch,loss.item())
            summary_writer.add_scalar('train_loss',loss, index)
            index+=1
        if epoch%100 == 0:
            torch.save(net.state_dict(), f'params/net{epoch}.pt')
            print(f'{epoch}保存成功')
        epoch+=1
