#!/usr/bin/env python
# encoding: utf-8

from PIL import Image
import os
import numpy as np
from guidedfilter import guidedfilter


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

if __name__ == '__main__':
    list_dir_file = os.walk("./images/")
    list_file = []
    for i in list_dir_file:
        list_file = i[2]

    for i in list_file:
        imageName = i
        file_path = os.path.join("./images/", imageName)
        Image.open(file_path).show()
        result = HazeRemoval(file_path)
        if not os.path.exists("./image_remove/"):
            print("sorry, it is not exist!")
            os.mkdir("./image_remove/")
        if os.path.exists("./image_remove/"):
            print("OK, it is exist!")

        file_processed = result.haze_removal()
        file_processed_path = os.path.join("./image_remove/", imageName)
        file_processed.save(file_processed_path)

        # with open(file_processed_path, 'w') as f_tem:
        #     f_tem.write(str(file_processed))

        file_processed.show()

