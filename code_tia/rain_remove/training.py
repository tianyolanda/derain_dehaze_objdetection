#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a re-implementation of training code of this paper:
# X. Fu, J. Huang, D. Zeng, Y. Huang, X. Ding and J. Paisley. “Removing Rain from Single Images via a Deep Detail Network”, CVPR, 2017.
# author: Xueyang Fu (fxy@stu.xmu.edu.cn)

import os
import h5py
import re
import numpy as np
import tensorflow as tf
import cv2
##################### select GPU device ####################################
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
############################################################################
#os.environ['CUDA_VISIBLE_DEVICES'] = str(monitoring_gpu.GPU_INDEX)
############################################################################

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_h5_file', 2000,
                            """number of training h5 files.""")
tf.app.flags.DEFINE_integer('num_patches', 500,
                            """number of patches in each h5 file.""")
tf.app.flags.DEFINE_integer('learning_rate', 0.1,
                            """learning rate.""")
tf.app.flags.DEFINE_integer('epoch', 60,
                            """epoch.""")
tf.app.flags.DEFINE_integer('batch_size', 20,
                            """Batch size.""")
tf.app.flags.DEFINE_integer('num_channels', 3,
                            """Number of channels of the input.""")
tf.app.flags.DEFINE_integer('image_size', 64,
                            """Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 64,
                            """Size of the labels.""")
tf.app.flags.DEFINE_string("data_path", "./data_generation/h5data/", "The path of h5 files")

tf.app.flags.DEFINE_string("save_model_path", "./model/", "The path of saving model")



# read h5 files
def read_data(file):
  with h5py.File(file, 'r') as hf:
    data = hf.get('data')
    label = hf.get('label')
    return np.array(data), np.array(label)



# guided filter
def guided_filter(data, num_patches = FLAGS.num_patches, width = FLAGS.image_size, height = FLAGS.image_size, channel = FLAGS.num_channels):
    r = 15
    eps = 1.0
    batch_q = np.zeros((num_patches, height, width, channel))
    for i in range(num_patches):
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



# initialize weights
def create_kernel(name, shape, initializer=tf.contrib.layers.xavier_initializer()):
    regularizer = tf.contrib.layers.l2_regularizer(scale = 1e-10)

    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer,
                                    regularizer=regularizer, trainable=True)
    return new_variables


# network structure
def inference(images, detail):

   #  layer 1
   with tf.variable_scope('conv_1'):
      kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels, 16])
      biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases_1')      
      scale = tf.Variable(tf.ones([16]), trainable=True, name='scale_1')
      beta = tf.Variable(tf.zeros([16]), trainable=True, name='beta_1')
  
      conv = tf.nn.conv2d(detail, kernel, [1, 1, 1, 1], padding='SAME')
      feature = tf.nn.bias_add(conv, biases)
  
      mean, var = tf.nn.moments(feature,[0, 1, 2])
      feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
  
      conv_shortcut = tf.nn.relu(feature_normal)
  
   #  layers 2 to 25
   for i in range(12):
     with tf.variable_scope('conv_%s'%(i*2+2)):
       kernel = create_kernel(name=('weights_%s'%(i*2+2)), shape=[3, 3, 16, 16])
       biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name=('biases_%s'%(i*2+2)))
       scale = tf.Variable(tf.ones([16]), trainable=True, name=('scale_%s'%(i*2+2)))
       beta = tf.Variable(tf.zeros([16]), trainable=True, name=('beta_%s'%(i*2+2)))   
   
       conv = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')     
       feature = tf.nn.bias_add(conv, biases)
   
       mean, var = tf.nn.moments(feature,[0, 1, 2])
       feature_normal = tf.nn.batch_normalization(feature,  mean, var, beta, scale, 1e-5)
   
       feature_relu = tf.nn.relu(feature_normal)

  
     with tf.variable_scope('conv_%s'%(i*2+3)): 
       kernel = create_kernel(name=('weights_%s'%(i*2+3)), shape=[3, 3, 16, 16])
       biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name=('biases_%s'%(i*2+3)))
       scale = tf.Variable(tf.ones([16]), trainable=True, name=('scale_%s'%(i*2+3)))
       beta = tf.Variable(tf.zeros([16]), trainable=True, name=('beta_%s'%(i*2+3)))   
   
       conv = tf.nn.conv2d(feature_relu, kernel, [1, 1, 1, 1], padding='SAME')     
       feature = tf.nn.bias_add(conv, biases)
  
       mean, var  = tf.nn.moments(feature,[0, 1, 2])
       feature_normal = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
   
       feature_relu = tf.nn.relu(feature_normal)  

       conv_shortcut = tf.add(conv_shortcut, feature_relu)  #  shortcut

  
   # layer 26
   with tf.variable_scope('conv_26'):
      kernel = create_kernel(name='weights_26', shape=[3, 3, 16, FLAGS.num_channels])   
      biases = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True, name='biases_26')
      scale = tf.Variable(tf.ones([3]), trainable=True, name=('scale_26'))
      beta = tf.Variable(tf.zeros([3]), trainable=True, name=('beta_26'))
  
      conv = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
      feature = tf.nn.bias_add(conv, biases)
 
      mean, var  = tf.nn.moments(feature,[0, 1, 2])
      neg_residual = tf.nn.batch_normalization(feature, mean, var, beta, scale, 1e-5)
 
      final_out = tf.add(images, neg_residual)

   return final_out
  


if __name__ == '__main__':

  images = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))  # data
  details = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels)) # label
  labels = tf.placeholder(tf.float32, shape=(None, FLAGS.label_size, FLAGS.label_size, FLAGS.num_channels))  # detail layer
  
  outputs = inference(images,details)
  
  loss = tf.reduce_mean(tf.square(labels - outputs))    # MSE loss

  
  lr_ = FLAGS.learning_rate 
  lr  = tf.placeholder(tf.float32 ,shape = []) 
  g_optim =  tf.train.AdamOptimizer(lr).minimize(loss) # Optimization method: Adam
  
  saver = tf.train.Saver(max_to_keep = 5)
  
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.5 # GPU setting
  config.gpu_options.allow_growth = True
  
  
  data_path = FLAGS.data_path
  save_path = FLAGS.save_model_path 
  epoch = int(FLAGS.epoch) 

  with tf.Session(config=config) as sess:

    sess.run(tf.global_variables_initializer())
    

    validation_data_name = "validation.h5"
    validation_h5_data, validation_h5_label = read_data(data_path + validation_data_name)


    validation_data = validation_h5_data
    validation_data = np.transpose(validation_data, (0,2,3,1))   # image data

    validation_detail = validation_data - guided_filter(validation_data)  # detail layer

    validation_label = np.transpose(validation_h5_label, (0,2,3,1)) # label


    if tf.train.get_checkpoint_state('./model/'):   # load previous trained model
      ckpt = tf.train.latest_checkpoint('./model/')
      saver.restore(sess, ckpt)
      ckpt_num = re.findall(r"\d",ckpt)
      if len(ckpt_num)==3:
        start_point = 100*int(ckpt_num[0])+10*int(ckpt_num[1])+int(ckpt_num[2])
      elif len(ckpt_num)==2:
        start_point = 10*int(ckpt_num[0])+int(ckpt_num[1])
      else:
        start_point = int(ckpt_num[0])      
      print("Load success")
   
    else:
      print("re-training")
      start_point = 0    


    for j in range(start_point,epoch):   # epoch

      if j+1 >(epoch/3):  # reduce learning rate
        lr_ = FLAGS.learning_rate*0.1
      if j+1 >(2*epoch/3):
        lr_ = FLAGS.learning_rate*0.01

      Training_Loss = 0.
  
      for num in range(FLAGS.num_h5_file):    # h5 files
        train_data_name = "train" + str(num+1) + ".h5"
        train_h5_data, train_h5_label = read_data(data_path + train_data_name)

        train_data = np.transpose(train_h5_data, (0,2,3,1))   # image data
        detail_data = train_data - guided_filter(train_data)  # detail layer
        train_label = np.transpose(train_h5_label, (0,2,3,1)) # label


        data_size = int( FLAGS.num_patches / FLAGS.batch_size )  # the number of batch
        for i in range(data_size):
          rand_index = np.arange(int(i*FLAGS.batch_size),int((i+1)*FLAGS.batch_size))   # batch
          batch_data = train_data[rand_index,:,:,:]   
          batch_detail = detail_data[rand_index,:,:,:]
          batch_label = train_label[rand_index,:,:,:]


          _,lossvalue = sess.run([g_optim,loss], feed_dict={images: batch_data, details: batch_detail, labels: batch_label, lr: lr_})
          Training_Loss += lossvalue  # training loss
  

      Training_Loss /=  (data_size * FLAGS.num_h5_file)

      model_name = 'model-epoch'   # save model
      save_path_full = os.path.join(save_path, model_name)
      saver.save(sess, save_path_full, global_step = j+1)

      Validation_Loss  = sess.run(loss,  feed_dict={images: validation_data, details: validation_detail, labels:validation_label})  # validation loss

      print ('%d epoch is finished, learning rate = %.4f, Training_Loss = %.4f, Validation_Loss = %.4f' %
               (j+1, lr_, Training_Loss, Validation_Loss))
  
