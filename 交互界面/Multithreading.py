"""
File: Multithreading.py
Author: CUMT-Muzihao
Date: 2024/05/23
Description:
"""
import threading
from time import time, sleep

import cv2

from Predict import YOLO

LEISURE = 1
BUSY = 0


class Multithread:
    def __init__(self, num, model_path):
        self.num = num
        self.list_in = []
        self.list_out = []
        model = YOLO(model_path)
        self.Units = [Unit(model) for _ in range(num)]
        self.Threads = [threading.Thread(target=unit, daemon=True).start() for unit in self.Units]
        self.order = []
        self.event = threading.Event()
        self.mainthread = threading.Thread(target=self.main_whlie, daemon=True).start()

    def __call__(self, image):
        while True:
            if len(self.list_in) <= self.num + 2:
                self.list_in.append(image)
                break

        if not len(self.list_out):
            while True:
                if len(self.list_out):
                    result = self.list_out[0]
                    break
            return result
        else:
            print(len(self.list_out))
            result = self.list_out[0]
            del self.list_out[0]
            return result

    def main_whlie(self):
        while True:
            if len(self.list_in):
                for i in range(self.num):
                    if self.Units[i].state() is LEISURE:
                        self.Units[i].load(self.list_in[0])
                        del self.list_in[0]
                        self.order.append(i)
                        break
            if len(self.order):
                if self.Units[self.order[0]].state() is LEISURE:
                    self.list_out.append(self.Units[self.order[0]].result)
                    del self.order[0]
            if self.event.is_set():
                break

    def termination(self):
        for unit in self.Units:
            unit.event.set()
        self.event.set()
        del self

    def __del__(self):
        return 0


class Unit:
    def __init__(self, model):
        self.image_in = None
        self.result = None
        self.model = model
        self.event = threading.Event()

    def __call__(self):
        while True:
            if self.image_in is not None:
                self.result = self.model(self.image_in)
                self.image_in = None

            if self.event.is_set():
                break

    def state(self):
        if self.image_in is None:
            return LEISURE
        else:
            return BUSY

    def load(self, image):
        self.image_in = image


# if __name__ == '__main__':
#     Model = Multithread(4, './asset/models/yolov8n.onnx')
#     cap = cv2.VideoCapture(0)
#     while True:
#         t = time()
#         _, image_in = cap.read()
#         # image = np.random.randint(0, 256, [400, 600, 3], np.uint8)
#         ret = Model(image_in)

#         # ret_img = ret.plot()
#         cv2.imshow('', image_in)
#         print(f"{1 / (time() - t):.2f}", end='\n')
#         if cv2.waitKey(5) == 27:
#             break
