from PIL import Image

# 图片缩放到416*416
def make_416_image(path):
    img=Image.open(path)
    w,h=img.size[0],img.size[1]
    temp=max(h,w)
    mask=Image.new(mode='RGB',size=(temp,temp),color=(0,0,0)) #填充黑度图
    mask.paste(img,(0,0))
    return mask