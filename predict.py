# coding:utf-8
# 对单张图片进行预测

from yolo import YOLO
from PIL import Image

yolo = YOLO()

img = '/home/bhap/Pytorch_test/YoloV3/img/1.jpg'
image = Image.open(img)
r_image = yolo.detect_image(image)
r_image.show()