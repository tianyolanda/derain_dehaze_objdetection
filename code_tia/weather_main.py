import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image as img
from PIL import ImageEnhance
import numpy as np
import cv2
from PIL import Image
from haze_remove.guidedfilter import guidedfilter
import matplotlib.image as mpimg
from SSD_tensorflow.nets import ssd_vgg_300, ssd_common, np_methods
from SSD_tensorflow.preprocessing import ssd_vgg_preprocessing
from SSD_tensorflow import visualization

slim = tf.contrib.slim


# 定义一个导向滤波器
def guided_filter(data, height, width):
    r = 15
    eps = 1.0
    batch_size = 1
    channel = 3
    batch_q = np.zeros((batch_size, height, width, channel))
    for i in range(batch_size):
        for j in range(channel):
            I = data[i, :, :, j]
            p = data[i, :, :, j]
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            mean_b = cv2.boxFilter(b, -1, (2 * r + 1, 2 * r + 1), normalize=False, borderType=0) / N
            q = mean_a * I + mean_b
            batch_q[i, :, :, j] = q
    return batch_q


def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-10)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables


# network structure
def inference(images, detail):
    #  layer 1
    with tf.variable_scope('conv_1'):
        kernel = create_kernel(name='weights_1', shape=[3, 3, 3, 16])
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases_1')
        scale = tf.Variable(tf.ones([16]), trainable=True, name='scale_1')
        beta = tf.Variable(tf.zeros([16]), trainable=True, name='beta_1')

        conv = tf.nn.conv2d(detail, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var = tf.nn.moments(feature, [0, 1, 2])
        feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        conv_shortcut = tf.nn.relu(feature_normal)

    # layers 2 to 25
    for i in range(12):
        with tf.variable_scope('conv_%s' % (i * 2 + 2)):
            kernel = create_kernel(name=('weights_%s' % (i * 2 + 2)), shape=[3, 3, 16, 16])
            biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True,
                                 name=('biases_%s' % (i * 2 + 2)))
            scale = tf.Variable(tf.ones([16]), trainable=True, name=('scale_%s' % (i * 2 + 2)))
            beta = tf.Variable(tf.zeros([16]), trainable=True, name=('beta_%s' % (i * 2 + 2)))

            conv = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
            feature = tf.nn.bias_add(conv, biases)

            mean, var = tf.nn.moments(feature, [0, 1, 2])
            feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

            feature_relu = tf.nn.relu(feature_normal)

        with tf.variable_scope('conv_%s' % (i * 2 + 3)):
            kernel = create_kernel(name=('weights_%s' % (i * 2 + 3)), shape=[3, 3, 16, 16])
            biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True,
                                 name=('biases_%s' % (i * 2 + 3)))
            scale = tf.Variable(tf.ones([16]), trainable=True, name=('scale_%s' % (i * 2 + 3)))
            beta = tf.Variable(tf.zeros([16]), trainable=True, name=('beta_%s' % (i * 2 + 3)))

            conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')
            feature = tf.nn.bias_add(conv, biases)

            mean, var = tf.nn.moments(feature, [0, 1, 2])
            feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

            feature_relu = tf.nn.relu(feature_normal)

            conv_shortcut = tf.add(conv_shortcut, feature_relu)  # shortcut

    # layer 26
    with tf.variable_scope('conv_26'):
        kernel = create_kernel(name='weights_26', shape=[3, 3, 16, 3])
        biases = tf.Variable(tf.constant(0.0, shape=[3], dtype=tf.float32), trainable=True,
                             name='biases_26')
        scale = tf.Variable(tf.ones([3]), trainable=True, name=('scale_26'))
        beta = tf.Variable(tf.zeros([3]), trainable=True, name=('beta_26'))

        conv = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
        feature = tf.nn.bias_add(conv, biases)

        mean, var = tf.nn.moments(feature, [0, 1, 2])
        neg_residual = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)

        final_out = tf.add(images, neg_residual)

    return final_out


def get_image_path():
    list_image = os.walk("./testimage/rain")
    list_image_path = []
    for i in list_image:
        list_image_path = i[2]
    return list_image_path


# 变成格式为:[:, :, :, :]的形式第一个维度为batch的size

# tensorflow当中的模型的加载方式和keras不同,tensorflow是加载模型时候,自动将对应的命名空间中的变量命名代替
# 从而生成一个一个训练好的模型. 重点是必须要定义一个完全一样的容器来进行变量的加载与存放.


def adjust_picture(image_raw):
    image_raw[np.where(image_raw < 0.)] = 0.
    image_raw[np.where(image_raw > 1.)] = 1.
    return image_raw


def rain_remove(image_path):
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

    with tf.Session(config=config) as sess:
        file_path = image_path
        ori = img.open(file_path)
        ori = ori.resize((500, 500))
        ori = np.array(ori)
        ori = ori / 255.0
        input = np.expand_dims(ori[:, :, :], axis=0)
        # 变成格式为:[:, :, :, :]的形式第一个维度为batch的size
        detail_layer = input - guided_filter(input, input.shape[1], input.shape[2])
        num_channels = 3
        image = tf.placeholder(tf.float32, shape=(1, input.shape[1], input.shape[2], num_channels))
        detail = tf.placeholder(tf.float32, shape=(1, input.shape[1], input.shape[2], num_channels))

        output = inference(image, detail)

        saver = tf.train.Saver()
        saver.restore(sess, "./rain_remove/model/test-model/model")

        print("rain remove load pre-trained model")
        final_output = sess.run(output, feed_dict={image: input, detail: detail_layer})

        final_output = adjust_picture(final_output)
        derained = final_output[0, :, :, :]

        derained = derained * 255
        derained_image = derained.astype(dtype="uint8")
        image_derained = img.fromarray(derained_image)
        # plt.show(image_derained)
        print("done!")
        return image_derained


class HazeRemoval:
    def __init__(self, filename, omega = 0.85, r = 40):
        self.filename = filename
        self.omega = omega
        self.r = r
        self.eps = 10 ** (-3)
        self.t = 0.1

    def _ind2sub(self, array_shape, ind):
        rows = (ind.astype('int') / array_shape[1])
        cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
        return (rows, cols)

    def _rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

    def haze_removal(self):
        oriImage = np.array(Image.open(self.filename))
        img = np.array(oriImage).astype(np.double) / 255.0
        grayImage = self._rgb2gray(img)

        darkImage = img.min(axis=2)

        (i, j) = self._ind2sub(darkImage.shape, darkImage.argmax())

        A = img[int(i), int(j), :].mean()
        transmission = 1 - self.omega * darkImage / A

        transmissionFilter = guidedfilter(grayImage, transmission, self.r, self.eps )
        transmissionFilter[transmissionFilter < self.t] = self.t

        resultImage = np.zeros_like(img)
        for i in range(3):
            resultImage[:, :, i] = (img[:, :, i] - A) / transmissionFilter + A

        resultImage[resultImage < 0] = 0
        resultImage[resultImage > 1] = 1
        result = Image.fromarray((resultImage * 255).astype(np.uint8))

        return result


def image_enhance(filename):
    image_brightness = ImageEnhance.Brightness(filename)
    final_picture = image_brightness.enhance(1.5)
    return final_picture


def SSD_fun(image_path):
    # 需要多少内存用多少, 不要一次性全部占用.
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    isess = tf.InteractiveSession(config=config)


    # Input placeholder.
    net_shape = (300, 300)
    data_format = 'NHWC'
    img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
    # Evaluation pre-processing: resize to SSD net shape.
    image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
        img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
    image_4d = tf.expand_dims(image_pre, 0)

    # Define the SSD model.
    reuse = True if 'ssd_net' in locals() else None
    ssd_net = ssd_vgg_300.SSDNet()
    with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
        predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

    # Restore SSD model. 这里有两个模型, 第一个模型稍微小一点, 准确度差百分之二.

    ckpt_filename = './SSD_tensorflow/checkpoints/ssd_300_vgg.ckpt'
    #ckpt_filename = './checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
    # ckpt_filename = '../checkpoints/VGG_VOC0712_SSD_300x300_ft_iter_120000.ckpt'
    isess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(isess, ckpt_filename)

    # SSD default anchor boxes.
    ssd_anchors = ssd_net.anchors(net_shape)


    '''
    ## Post-processing pipeline

    The SSD outputs need to be post-processed to provide proper detections. Namely, we follow these common steps:

    * Select boxes above a classification threshold;
    * Clip boxes to the image shape;
    * Apply the Non-Maximum-Selection algorithm: fuse together boxes whose Jaccard score > threshold;
    * If necessary, resize bounding boxes to original image shape.
    '''


    # Main image processing routine.
    def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
        # Run SSD network.
        rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                                  feed_dict={img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes

    image_names = image_path

    img = mpimg.imread(image_names)
    rclasses, rscores, rbboxes =  process_image(img)
    image_back = visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    return image_back



















if __name__ =="__main__":
    #image_path = "./testimage/rain/6.jpg"
    image_path = "./testimage/mix/1.jpg"
    # #下面是去雨处理
    # image_processed = rain_remove(image_path)
    # image_processed.show()
    # image_processed.save("./testimage/tem/tem.jpg")
    # image_path = "./testimage/tem/tem.jpg"

    # #下面是去雾处理
    # Image.open(image_path)
    # result = HazeRemoval(image_path)
    # file_processed = result.haze_removal()
    # file_processed = image_enhance(file_processed)
    # file_processed.show()
    # file_processed.save("./testimage/tem/tem.jpg")
    # image_path = "./testimage/tem/tem.jpg"

    # 这里是ssd
    image_processed = SSD_fun(image_path)

