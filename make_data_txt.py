"""
解析xml信息到data.txt
"""

import math
import xml.etree.ElementTree as et
import os
from PIL import Image
class_num={
    'person':0,
    'horse':1,
    'bicycle':2,
}
script_dir = os.path.dirname(os.path.realpath(__file__))
xml_dir = os.path.join(script_dir, 'data/image_voc')
xml_filenames=os.listdir(xml_dir) # 获取文件夹下的所有文件名

with open(script_dir+'/data.txt','a') as f:
    for xml_filename in xml_filenames:
        xml_filename_path = os.path.join(xml_dir,xml_filename)
        tree = et.parse(xml_filename_path) # 获取xml信息
        root = tree.getroot()
        filename = root.find('filename')
        names = root.findall('object/name')
        box=root.findall('object/bndbox')
        #for x1,y1,x2,y2 in box:
            #print(x1.text)
        data=[]
        data.append(filename.text)
        for name,box in zip(names,box):
            cls = class_num[name.text]
            # math.floor向下取整
            cx,cy=math.floor((int(box[0].text)+int(box[2].text))/2),math.floor((int(box[1].text)+int(box[3].text))/2)
            w,h=(int(box[2].text)-int(box[0].text)),(int(box[3].text)-int(box[1].text))
            data.append(cls)
            data.append(cx)
            data.append(cy)
            data.append(w)
            data.append(h)
        _str=''
        for i in data:
            _str=_str+str(i)+' '
        f.write(_str+'\n') 
f.close()




