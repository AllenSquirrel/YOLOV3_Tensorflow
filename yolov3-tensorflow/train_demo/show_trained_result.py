import tensorflow as tf
from core import utils, yolov3
import cv2
from PIL import Image
import numpy as np

input_image = "../data/train_dome_data/images/raccoon-4.jpg"
image = Image.open(input_image)
# image = cv2.imread(input_image)
# image = Image.fromarray(image)
image_resize = cv2.resize(np.array(image) / 255., (416, 416))
image_place = tf.placeholder(dtype=tf.float32, shape=(None, 416, 416, 3))
CLASSES = utils.read_coco_names('../data/raccoon.names')
ANCHORE = utils.get_anchors("../data/raccoon_anchors.txt", 416, 416)
model = yolov3.yolov3(len(CLASSES), ANCHORE)
with tf.variable_scope('yolov3'):
    pred_feature_map = model.forward(image_place, is_training=False)#执行前向传播得到3个尺度的feature map
    pred = model.predict(pred_feature_map)#根据feature map进行预测 
sess = tf.Session()
saver = tf.train.Saver()
model_dir = tf.train.latest_checkpoint("../data/train_dome_data/model/")
saver.restore(sess, model_dir)
boxes, confs, prods = sess.run(pred, feed_dict={image_place: np.expand_dims(image_resize, 0)})#根据预测结果和获取目标预测框，目标类别 以及是否有object
boxes, confs, prods = utils.cpu_nms(boxes, confs * prods, len(CLASSES))#根据非最大值抑制算法的原则  筛选出最优的候选框
utils.draw_boxes(image, boxes, confs, prods, CLASSES, (416, 416), "../data/font/HuaWenXinWei-1.ttf")#在输出图片上绘制目标框，并输出标签识别率结果
print(boxes, confs, prods)
