import cv2
import tensorflow as tf
import matplotlib.image as mpimg
from SSD_tensorflow.nets import ssd_vgg_300, ssd_common, np_methods
from SSD_tensorflow.preprocessing import ssd_vgg_preprocessing
from SSD_tensorflow import visualization
import random
import PIL as plt
import time
import numpy as np

slim = tf.contrib.slim


class SSD_entity:
    def __init__(self):
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.isess = tf.InteractiveSession(config=config)

        # Input placeholder.
        self.net_shape = (300, 300)
        self.data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        # Evaluation pre-processing: resize to SSD net shape.
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, self.net_shape, self.data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)

        # Define the SSD model.
        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=self.data_format)):
            self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)


        ckpt_filename = './SSD_tensorflow/checkpoints/ssd_300_vgg.ckpt'
        # ckpt_filename = './checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, ckpt_filename)

        # SSD default anchor boxes.
        self.ssd_anchors = ssd_net.anchors(self.net_shape)

    def process_image(self, img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, self.ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes


def plt_bboxes(img, classes, scores, bboxes, figsize=(10,10), linewidth=3):
    # img = np.array(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax),
                                 color=colors[cls_id],
                                 thickness=linewidth)
            class_name = str(cls_id)
            font = cv2.FONT_HERSHEY_TRIPLEX
            cv2.putText(frame, '{:s} | {:.3f}'.format(class_name, score), (xmin, ymin+2), font, 0.5, (255, 255, 255), thickness=1)
    return frame


if __name__ == "__main__":
    ssd_entity = SSD_entity()

    vedio_path = "./video/1.mp4"
    cap = cv2.VideoCapture(vedio_path)

    while(1):
        # time.sleep(0.1)
        try:
            ret, frame = cap.read()
            print(frame)
            rclasses, rscores, rbboxes = ssd_entity.process_image(frame)
            # plt_back = visualization.plt_bboxes(frame, rclasses, rscores, rbboxes)
            image_back = plt_bboxes(frame, rclasses, rscores, rbboxes)
            cv2.imshow("image_back", image_back)
            if cv2.waitKey(3) & 0xFF == ord('q'):
                break
        except:
            print("OVER")
            break



