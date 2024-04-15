# 注释
DATA_HEIGHT = 416
DATA_WIDTH = 416

CLASS_NUM = 3

anchors = {
    13: [[270, 254], [291, 179], [162, 304]],
    26: [[175, 222], [112, 235], [175, 140]],
    52: [[81, 118], [53, 142], [44, 28]]
}

ANCHORS_AREA = {
    13: [x * y for x, y in anchors[13]],
    26: [x * y for x, y in anchors[26]],
    52: [x * y for x, y in anchors[52]],
}
