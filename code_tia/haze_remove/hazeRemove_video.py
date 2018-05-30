import cv2
from PIL import Image, ImageEnhance
import numpy as np
from guidedfilter import guidedfilter


cap = cv2.VideoCapture("./test_video.mp4")
fourcc = cv2.VideoWriter_fourcc("D", "I", "B", " ")
out = cv2.VideoWriter('output.mp4', fourcc, 20, (640, 352))

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
        #oriImage = np.array(Image.open(self.filename))
        oriImage = np.array(self.filename)
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
       # result = Image.fromarray((resultImage * 255).astype(np.uint8))
        result = (resultImage * 255).astype(np.uint8)
        print("success!")

        return result

while True:
    ret, frame = cap.read()

    result = HazeRemoval(frame)
    print(frame.shape)

    frame_haze = result.haze_removal()

    image_tem = Image.fromarray(frame_haze)
    image_brightness = ImageEnhance.Brightness(image_tem)
    final_picture = image_brightness.enhance(2.0)
    # image_contrast = ImageEnhance.Contrast(final_picture)
    # final_picture = image_contrast.enhance(1.5)
    final_fram = np.array(final_picture)

    cv2.imshow("haze_beijing", final_fram)
    # 这里是对waitKey的一个理解,等待时间中返回按键的ASCII.


    if cv2.waitKey(10) == ord("q"):
        break


cv2.destroyAllWindows()