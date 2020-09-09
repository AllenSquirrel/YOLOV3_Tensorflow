import os
import sys
import wget
import time
import argparse
import tensorflow as tf
from core import yolov3, utils

#参数解析 
class parser(argparse.ArgumentParser):

    def __init__(self, description):
        super(parser, self).__init__(description)
        #获取训练后保存的模型
        self.add_argument(
            "--ckpt_file", "-cf", default='./checkpoint/yolov3.ckpt', type=str,
            help="[default: %(default)s] The checkpoint file ...",
            metavar="<CF>",
        )
        #获取类别 默认coco数据集80分类
        self.add_argument(
            "--num_classes", "-nc", default=80, type=int,
            help="[default: %(default)s] The number of classes ...",
            metavar="<NC>",
        )
        #获取yolov3随机发布的候选框 anchorbox 每一尺度发布三个候选框
        self.add_argument(
            "--anchors_path", "-ap", default="./data/coco_anchors.txt", type=str,
            help="[default: %(default)s] The path of anchors ...",
            metavar="<AP>",
        )
        #获取训练后得到的权重文件 
        self.add_argument(
            "--weights_path", "-wp", default='./checkpoint/yolov3.weights', type=str,
            help="[default: %(default)s] Download binary file with desired weights",
            metavar="<WP>",
        )
        #将获取权重文件 解析并转换到对应网络结构
        self.add_argument(
            "--convert", "-cv", action='store_false',
            help="[default: %(default)s] Downloading yolov3 weights and convert them",
        )
        #使用迁移学习 加载预训练模型冻结前端参数  只对后续参数权重进行更新训练
        self.add_argument(
            "--freeze", "-fz", action='store_false',
            help="[default: %(default)s] freeze the yolov3 graph to pb ...",
        )
        #输入图片高
        self.add_argument(
            "--image_h", "-ih", default=416, type=int,
            help="[default: %(default)s] The height of image, 416 or 608",
            metavar="<IH>",
        )
        #输入图片宽
        self.add_argument(
            "--image_w", "-iw", default=416, type=int,
            help="[default: %(default)s] The width of image, 416 or 608",
            metavar="<IW>",
        )
        #给定IOU阈值  默认0.5  一般调节范围在0.5-0.7
        self.add_argument(
            "--iou_threshold", "-it", default=0.5, type=float,
            help="[default: %(default)s] The iou_threshold for gpu nms",
            metavar="<IT>",
        )
        #给定识别率阈值 默认0.5  50%以上的识别率认为基本满足目标检测要求
        self.add_argument(
            "--score_threshold", "-st", default=0.5, type=float,  # 分数阈值
            help="[default: %(default)s] The score_threshold for gpu nms",
            metavar="<ST>",
        )


def main(argv):
    flags = parser(description="freeze yolov3 graph from checkpoint file").parse_args()
    print("=> the input image size is [%d, %d]" % (flags.image_h, flags.image_w))
    anchors = utils.get_anchors(flags.anchors_path, flags.image_h, flags.image_w)
    # print(anchors)
    # exit()
    model = yolov3.yolov3(flags.num_classes, anchors)

    with tf.Graph().as_default() as graph:
        sess = tf.Session(graph=graph)
        inputs = tf.placeholder(tf.float32, [1, flags.image_h, flags.image_w, 3])  # 输入占位符 参数后传递
        print("=>", inputs)

        with tf.variable_scope('yolov3'):
            feature_map = model.forward(inputs, is_training=False)  # 返回3个尺度的feature_map

        # 获取网络给出绝对boxes(左上角,右下角)信息, 
        boxes, confs, probs = model.predict(feature_map)
        scores = confs * probs     # scores = confs * probs
        print("=>", boxes.name[:-2], scores.name[:-2])
        # cpu 运行是恢复模型所需要的网络节点的名字
        cpu_out_node_names = [boxes.name[:-2], scores.name[:-2]]
        boxes, scores, labels = utils.gpu_nms(boxes, scores, flags.num_classes,
                                              score_thresh=flags.score_threshold,
                                              iou_thresh=flags.iou_threshold)
        print("=>", boxes.name[:-2], scores.name[:-2], labels.name[:-2])
        # gpu 运行是恢复模型所需要的网络节点的名字 , 直接运算得出最终结果
        gpu_out_node_names = [boxes.name[:-2], scores.name[:-2], labels.name[:-2]]
        feature_map_1, feature_map_2, feature_map_3 = feature_map
        saver = tf.train.Saver(var_list=tf.global_variables(scope='yolov3'))
        if flags.convert:#对于已有模型及对应权重文件进行加载并转换
            if not os.path.exists(flags.weights_path):
                url = 'https://github.com/YunYang1994/tensorflow-yolov3/releases/download/v1.0/yolov3.weights'
                for i in range(3):
                    time.sleep(1)
                    print("=> %s does not exists ! " % flags.weights_path)
                print("=> It will take a while to download it from %s" % url)
                print('=> Downloading yolov3 weights ... ')
                wget.download(url, flags.weights_path)

            load_ops = utils.load_weights(tf.global_variables(scope='yolov3'), flags.weights_path)
            sess.run(load_ops)
            save_path = saver.save(sess, save_path=flags.ckpt_file)
            print('=> model saved in path: {}'.format(save_path))

        # print(flags.freeze)
        if flags.freeze:#加载冻结网络模型（预训练模型）
            saver.restore(sess, flags.ckpt_file)
            print('=> checkpoint file restored from ', flags.ckpt_file)
            utils.freeze_graph(sess, './checkpoint/yolov3_cpu_nms.pb', cpu_out_node_names)
            utils.freeze_graph(sess, './checkpoint/yolov3_gpu_nms.pb', gpu_out_node_names)


if __name__ == "__main__": main(sys.argv)
