import sys
sys.path.append("/usr/lib/python3.7/")
#import tensorflow as tf

import platform
import cv2
from threading import Thread
from video_stream import VideoStream
from datetime import datetime
import datetime
import time
import numpy as np
import os
from detector import Detector
from picamera.array import PiRGBArray
from picamera import PiCamera
from database_manager import DatabaseManager
import pymongo
import ast
import tensorflow as tf
import nms

SSD_MOBILENET_V2_SAVED_MODEL_PATH="/home/pi/Desktop/My_Server/backend/models/"
category_map = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}
img_width=300
img_height=300


class DetectionManager:
    def __init__(self):
        self.model_paths = {"SSD_Mobilenet_v2_320x320":"/home/pi/Desktop/My_Server/backend/models/"}
        self.category_maps = {"SSD_Mobilenet_v2_320x320":{1: "person"}}

    def setupDetection(self, detection_period_id, neuralNetworkType="SSD_Mobilenet_v2_320x320", detectionSeconds=60,  obj_threshold=0.3,
                       video_resolution={"width":320, "height":320}, framerate=30):
        self.neuralNetworkType = neuralNetworkType
        self.obj_threshold = obj_threshold
        self.detectionSeconds = detectionSeconds
        self.video_resolution = video_resolution
        self.framerate = framerate
        #self.detector = Detector(self.model_paths["SSD_Mobilenet_v2_320x320"], self.category_maps["SSD_Mobilenet_v2_320x320"])
        self.database_manager = DatabaseManager()
        self.detection_period_id = detection_period_id
        print("Starting detection: " + str(datetime.datetime.now()))

        detections = self.detect()
        print("Ending detection: " + str(datetime.datetime.now()))

    def detect(self):
        detection_results = None

        xml_file = '/home/pi/Desktop/My_Server/backend/endpoints/haarcascades/haarcascade_frontalface_default.xml'
        
        classifier = cv2.CascadeClassifier(xml_file)
        # Otherwise, grab the next frame from the stream
        frame = cv2.imread('/home/pi/Desktop/twarz.jpg')
        
        height, width, channels = frame.shape
        
        resolution={"width":width, "height":height}

        #detection_results = self.detector.detect(frame, resolution["width"], resolution["height"])
        
        
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces_rect = classifier.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=9)
        
        print(str(detection_results))
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
        #print("How many detected boxes were created: " + str(len(detection_results['detection_boxes'][0].numpy())))
        
        #selected_boxes = nms.non_max_suppression_fast(detection_results['detection_boxes'][0].numpy(), 0.4)
        '''
        selected_boxes = detection_results['detection_boxes'][0]
        
        for box, score, det_class in zip(detection_results['detection_boxes'][0], detection_results['detection_scores'][0], detection_results['detection_classes'][0]):
             if score > 0.2 and int(det_class) == 1: 
                frame = cv2.rectangle(frame, (int(box[1] * resolution["width"]), int(box[0] * resolution["height"])),
                                      (int(box[3] * resolution["width"]), int(box[2] * resolution["height"])), (255, 0, 0), 2)
                                      '''
        '''
             elif score > 0.2 and int(det_class) != 1:
                 frame = cv2.rectangle(frame, (int(box[1] * resolution["width"]), int(box[0] * resolution["height"])),
                      (int(box[3] * resolution["width"]), int(box[2] * resolution["height"])), (0, 255, 0), 2)
        '''

        cv2.imshow("Actual frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
dm = DetectionManager()

dm.setupDetection(1000)

dm.detect()
