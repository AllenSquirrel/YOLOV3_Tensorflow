import cv2
import numpy as np
import tensorflow as tf
from core import utils
from PIL import Image
from core.dataset import Parser, dataset

sess = tf.Session()

IMAGE_H, IMAGE_W = 416, 416
BATCH_SIZE = 1
SHUFFLE_SIZE = 1

train_tfrecord = "../data/train_dome_data/images_train.tfrecords"
anchors = utils.get_anchors('../data/raccoon_anchors.txt', IMAGE_H, IMAGE_W)
classes = utils.read_coco_names('../data/raccoon.names')
# print(classes)
num_classes = len(classes)  # 识别的种类

parser = Parser(IMAGE_H, IMAGE_W, anchors, num_classes, debug=True)
trainset = dataset(parser, train_tfrecord, BATCH_SIZE, shuffle=SHUFFLE_SIZE)

is_training = tf.placeholder(tf.bool)#此处占位符  方便session后续传参 
example = trainset.get_next()#根据batch_size依次加载数据集，放入网络进行训练 

for l in range(1):
    image, boxes = sess.run(example)
    # print(image)
    # print(sess.run(example))
    image, boxes = image[0], boxes[0]

    n_box = len(boxes)
    for i in range(n_box):
        image = cv2.rectangle(image, (int(float(boxes[i][0])),
                                      int(float(boxes[i][1]))),
                              (int(float(boxes[i][2])),
                               int(float(boxes[i][3]))), (255, 0, 0), 1)#绘制目标框
        label = classes[boxes[i][4]]
        image = cv2.putText(image, label, (int(float(boxes[i][0])), int(float(boxes[i][1]))),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)#标注 绿色文字类别标签

    image = Image.fromarray(np.uint8(image))
    image.show()
