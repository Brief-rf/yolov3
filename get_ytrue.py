import glob

import numpy as np
import os
from PIL import Image
from get_xml import PascalVocXmlParser


def get_parse(ann_fname,input_size):
    parser = PascalVocXmlParser()
    fname = parser.get_fname(ann_fname)
    weight = parser.get_width(ann_fname)
    height = parser.get_height(ann_fname)
    labels = parser.get_labels(ann_fname)
    boxes = parser.get_boxes(ann_fname)

    for i in range(len(boxes)):
        boxes[i][0] = boxes[i][0]/weight*input_size
        boxes[i][1] = boxes[i][1]/weight*input_size
        boxes[i][2] = boxes[i][2]/height*input_size
        boxes[i][3] = boxes[i][3]/height*input_size

    return fname,labels,boxes

def get_IOU(box1,box2):
    w_min = min(box1[1],box2[1])
    h_min = min(box1[3],box2[3])
    w = w_min - box1[0]
    h = h_min - box1[2]

    intersect = w * h
    merge = (box1[1] - box1[0]) * (box1[3] - box1[2]) + (box2[1] - box2[0]) * (box2[3] - box2[2])
    IOU = intersect/(merge-intersect)

    return IOU

def get_anchor(anchors,box):
    IOUlist = []
    anchorslist = np.zeros((len(anchors), 4), dtype='float32')
    for i in range(len(anchorslist)):
        anchorslist[i][0] = box[0]
        anchorslist[i][1] = anchors[i][0] + anchorslist[i][0]
        anchorslist[i][2] = box[2]
        anchorslist[i][3] = anchors[i][1] + anchorslist[i][2]

        IOU = get_IOU(box,anchorslist[i])
        IOUlist.append(IOU)

    anchor = IOUlist.index((max(IOUlist)))

    return anchor

def get_ytrue(boxes,anchors,anchor_shape,b,pattern_shape,input_size,classes,labels,ytrues):
    #初始化newbox为全零数组
    newbox = np.zeros((4), dtype='float32')
    #遍历一张图中所有boxes
    for i in range(len(boxes)):
        #获取与box iou最大的anchor的序号
        anchor = get_anchor(anchors,boxes[i])
        #根据序号确定box于9种特征图的位置
        layer_anchor = anchor//anchor_shape[1]
        box_anchor = anchor%anchor_shape[1]
        #将box的尺寸按特征图尺寸进行缩放，并计算中心点的坐标
        rate = pattern_shape[layer_anchor] / input_size
        cent_x = (boxes[i][0] + boxes[i][1]) / 2 * rate
        cent_y = (boxes[i][2] + boxes[i][3]) / 2 * rate
        #计算中心点网格坐标
        x = np.floor(cent_x).astype('int32')#x代表列数
        y = np.floor(cent_y).astype('int32')#y代表行数
        w = boxes[i][1] - boxes[i][0]
        h = boxes[i][3] - boxes[i][2]
        #获取标签序号
        c = classes.index(labels[i])
        #newbox中的w、h尺寸是关于特征图的
        newbox[0] = cent_x
        newbox[1] = cent_y
        newbox[2] = np.log(max(w,1) / anchors[anchor][0])
        newbox[3] = np.log(max(h,1) / anchors[anchor][1])
        #更新ytrue(由于每种特征图尺度不同，故通过layer_anchor区分赋值)
        ytrues[layer_anchor][b, y, x, box_anchor,0:4] = newbox[0:4]
        ytrues[layer_anchor][b, y, x, box_anchor, 4] = 1
        ytrues[layer_anchor][b, y, x, box_anchor, 5 + c] = 1

    return ytrues

def get_img(img_dir,fname,input_size):
    img_fname = os.path.join(img_dir,fname)
    image = Image.open(img_fname)
    #原则上应以灰色像素填充至方形
    image = image.resize((input_size,input_size))
    image = np.array(image, dtype='float32')
    #数值归一化
    image /= 255.

    return image

#数据生成
def generator(batch_size,pattern_shape,anchor_shape,classes,ann_fnames,input_size,anchors,img_dir):
    n = len(ann_fnames)
    i = 0
    #循环返回inputs和ytrues
    while True:
        inputs = []
        # 初始化ytrues为全0数组,依次为小、中、大尺寸
        #
        ytrues = [np.zeros((batch_size,pattern_shape[l],pattern_shape[l],anchor_shape[1],5+len(classes)))
                  for l in range(anchor_shape[0])]

        for b in range(batch_size):
            # 随机打乱标签文件名列表
            if i == 0:
                np.random.shuffle(ann_fnames)
            # 获取文件名、标签、box（xmin,xmax,ymin,ymax）(尺寸按inputsize比例缩放)
            fname,labels,boxes = get_parse(ann_fnames[i],input_size)
            #更新ytures
            ytrues = get_ytrue(boxes,anchors,anchor_shape,b,pattern_shape,input_size,classes,labels,ytrues)
            #获取input
            img = get_img(img_dir,fname,input_size)
            inputs.append(img)
            i = (i+1)%n
        inputs = np.array(inputs)
        #ytrues特征图尺寸依次为大、中、小
        yield inputs,[ytrues[2],ytrues[1],ytrues[0]]


def main():
    root = os.path.dirname(__file__)
    ann_dir = os.path.join(root, "dataSet", "xml", "*.xml")
    ann_fnames = glob.glob(ann_dir)
    print(ann_fnames)
    img_dir = os.path.join(root, "dataSet", "img")
    anchros = np.array([[8, 73], [8, 24], [13, 32], [19, 51], [35, 64], [25, 37], [22, 164], [95, 195], [57, 104]])
    classes = ["flower"]
    input_size = 416
    batch_size = 1
    pattern_shape = [52, 26, 13]
    anchro_shape = [3, 3]
    for inputs, ytrues in generator(batch_size, pattern_shape, anchro_shape, classes, ann_fnames, input_size, anchros, img_dir):
        print(inputs.shape, ytrues[0].shape, ytrues[1].shape, ytrues[2].shape)
        print(len(inputs), len(ytrues))

if __name__ == '__main__':
    main()