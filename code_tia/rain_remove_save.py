import os
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image as img
from PIL import ImageEnhance
import numpy as np
import cv2
from PIL import Image
import os
from haze_remove.guidedfilter import guidedfilter


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
    with tf.Session() as sess:
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
    final_picture = image_brightness.enhance(2.5)
    return final_picture


if __name__ =="__main__":
    #image_path = "./testimage/rain/6.jpg"
    image_path = "./testimage/haze/7.jpg"
    # image_processed = rain_remove(image_path)
    # image_processed.show()

    # Image.open(image_path).show()
    # result = HazeRemoval(image_path)
    # file_processed = result.haze_removal()
    # file_processed = image_enhance(file_processed)
    # file_processed.show()