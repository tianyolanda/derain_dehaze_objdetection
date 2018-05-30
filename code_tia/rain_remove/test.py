import os
import tensorflow as tf
import matplotlib.pyplot as plt
import PIL.Image as img
import numpy as np

image_path = "./image/rain.jpg"
# 记录一下,imread读到的是一个numpy的数组,而pil读到的是image对象.
image_1 = img.open(image_path)
image_1 = image_1.resize((576, 480))
image_array = np.array(image_1)
print(image_array.shape)
image_reco = img.fromarray(image_array, mode="RGB")
image_reco.show()

# print("image_1.shape:  ", image.shape)
# plt.imshow(image_array)
# plt.show()