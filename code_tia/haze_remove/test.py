#!/usr/bin/env python
# encoding: utf-8

from PIL import Image
from matplotlib import pylab as plt_figure
import numpy as np
import os


def haze_removal(image, w0=0.6, t0=0.1):
    # density:描述雾的浓度, density越大,则越浓.
    density = 2
    darkImage = image.min(axis=2)
    maxDarkChannel = darkImage.max()
    t = 1 / (density - density*(darkImage/maxDarkChannel) + 1)
    J = image
    # J[:, :, 0] = image[:, :, 0] * t + maxDarkChannel*(1 - t)
    # J[:, :, 1] = image[:, :, 1] * t + maxDarkChannel*(1 - t)
    # J[:, :, 2] = image[:, :, 2] * t + maxDarkChannel*(1 - t)
    J[:, :, 0] = image[:, :, 0] * t + maxDarkChannel*(1 - t)
    J[:, :, 1] = image[:, :, 1] * t + maxDarkChannel*(1 - t)
    J[:, :, 2] = image[:, :, 2] * t + maxDarkChannel*(1 - t)
    print(maxDarkChannel)
    return Image.fromarray(J)

    # maxDarkChannel = darkImage.max()
    # darkImage = darkImage.astype(np.double)
    #
    # t = 1 - w0 * (darkImage / maxDarkChannel)
    # # T = t * 255
    # # T.dtype = 'uint8'
    #
    # t[t < t0] = t0
    #
    # J = image
    # J[:, :, 0] = (image[:, :, 0] - (1 - t) * maxDarkChannel) / t
    # J[:, :, 1] = (image[:, :, 1] - (1 - t) * maxDarkChannel) / t
    # J[:, :, 2] = (image[:, :, 2] - (1 - t) * maxDarkChannel) / t
    # result = Image.fromarray(J)
    #
    # return result


if __name__ == '__main__':
    list_dir_file = os.walk("./Image_test/")
    list_file = []
    for i in list_dir_file:
        list_file = i[2]

    for i in list_file[1:]:
        imageName = i
        file_path = os.path.join("./Image_test", imageName)
        Image.open(file_path).show()
        image_file = np.array(Image.open(file_path))
        image_size = image_file.shape
        result = haze_removal(image_file)
        result.show()
        plt_figure.show()
