# coding:utf-8
from __future__ import division
import time
import numpy as np
import cv2
from PIL import Image, ImageFont, ImageDraw
from yolo import YOLO
from utils.sort import sort_image, Sort
from collections import deque

pts = [deque(maxlen=30) for _ in range(9999)]
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")

mp4 = cv2.VideoCapture('/home/bhap/Documents/Video/test3.MP4')
model_path = '/home/bhap/Pytorch_test/YoloV3_Sort/model_data/bdd.pth'
CAMERA = False

def main(yolo):

    capture = cv2.VideoCapture(0)
    tracker = Sort(max_age=10, min_hits=3)  # 存储的帧数和连续关联的帧数
    fps = 0.0

    while True:

        t1 = time.time()

        if CAMERA:
            ref, frame = capture.read()
        else:
            ref, frame = mp4.read()
        if ref != True:
            break

        # 格式转变 BGR2RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        boxs, classes = yolo.sort_detect_image(frame)

        out_boxes, out_classes, object_id = sort_image(tracker, boxs, classes)

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(2e-2 * frame.size[1] + 0.5).astype('int32'))
        thickness = (frame.size[0] + frame.size[1]) // 800

        for i, c in enumerate(out_classes):
            predicted_class = yolo.class_names[c]
            id = int(object_id[i])
            left, top, right, bottom = out_boxes[i]
            color = tuple([int(k) for k in COLORS[id % len(COLORS)]])

            label = '{} id:{}'.format(predicted_class, id)
            draw = ImageDraw.Draw(frame)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for j in range(thickness):
                draw.rectangle(
                    [left + j, top + j, right - j, bottom - j], outline=color)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)], fill=color)
            draw.text(text_origin, str(label), fill=(0, 0, 0), font=font)
            del draw

        frame = np.array(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fps = (fps + (1. / (time.time() - t1))) / 2
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('mp4', frame)

        cv = cv2.waitKey(30) & 0xff

        if cv == 27:
            capture.release()
            break

if __name__ == '__main__':
    main(YOLO())