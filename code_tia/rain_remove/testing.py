# -*- coding: utf-8 -*-


# This is a re-implementation of testing code of this paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)




import os
import training as Network
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
import cv2

##################### Select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################
#os.environ['CUDA_VISIBLE_DEVICES'] = str(monitoring_gpu.GPU_INDEX)
############################################################################


        
def guided_filter(data, height,width):
    r = 15
    eps = 1.0
    batch_size = 1
    channel = 3
    batch_q = np.zeros((batch_size, height, width, channel))
    for i in range(batch_size):
        for j in range(channel):
            I = data[i, :, :,j] 
            p = data[i, :, :,j] 
            ones_array = np.ones([height, width])
            N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
            mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            cov_Ip = mean_Ip - mean_I * mean_p
            mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            var_I = mean_II - mean_I * mean_I
            a = cov_Ip / (var_I + eps) 
            b = mean_p - a * mean_I
            mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
            q = mean_a * I + mean_b 
            batch_q[i, :, :,j] = q 
    return batch_q


file = "./image/rain.jpg"
ori = img.imread(file)
ori = ori/255.0

input = np.expand_dims(ori[:,:,:], axis = 0)
detail_layer = input - guided_filter(input, input.shape[1], input.shape[2])


num_channels = 3
image = tf.placeholder(tf.float32, shape=(1, input.shape[1], input.shape[2], num_channels))
detail = tf.placeholder(tf.float32, shape=(1, input.shape[1], input.shape[2], num_channels))

output = Network.inference(image, detail)

saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    
    if tf.train.get_checkpoint_state('./model/'):  
        ckpt = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, ckpt)
        print ("Loading model")

    else:
        saver.restore(sess, "./model/test-model/model") # please note this testing model is for debug only and not completed trained
        print ("load pre-trained model")


    final_output  = sess.run(output, feed_dict={image:input, detail: detail_layer})

    final_output[np.where(final_output < 0. )] = 0.
    final_output[np.where(final_output > 1. )] = 1.
    derained = final_output[0,:,:,:]
        
    plt.subplot(1,2,1)     
    plt.imshow(ori)      
    plt.title('input')

    plt.subplot(1,2,2)    
    plt.imshow(derained)
    plt.title('output')

    plt.show()
