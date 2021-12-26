import sys
sys.path.append("/usr/lib/python3.7/")
import platform
import cv2
from threading import Thread
from video_stream import VideoStream
from datetime import datetime
import datetime
import time
import numpy as np
import os
from detector_cv2 import Detector
from picamera.array import PiRGBArray
from picamera import PiCamera
from database_manager import DatabaseManager
import pymongo
import ast
import tensorflow as tf
import nms


class DetectionManager:
    def __init__(self):
        self.model_paths = {"SSD_Mobilenet_v2_320x320":"/home/pi/Desktop/My_Server/backend/models/"}
        self.category_maps = {"SSD_Mobilenet_v2_320x320":{1: "person"}}

    def setupDetection(self, detection_period_id, neuralNetworkType="SSD_Mobilenet_v2_320x320", detectionSeconds=60,  obj_threshold=0.3,
                       video_resolution={"width":320, "height":320}, framerate=30):
        #TU SPRAWDZIC
        self.neuralNetworkType = neuralNetworkType
        self.obj_threshold = obj_threshold
        self.detectionSeconds = detectionSeconds
        self.video_resolution = video_resolution
        self.framerate = framerate
        self.detector = Detector("SSD_Mobilenet_v2_320x320")
        self.database_manager = DatabaseManager()
        #print("In setupDetection: " + str(detection_period_id))
        self.detection_period_id = detection_period_id
        #print("Starting detection: " + str(datetime.datetime.now()))
        detections = self.detect()
        #print("Ending detection: " + str(datetime.datetime.now()))

    def detect(self):
        resolution={"width":1200, "height":720}
        framerate=30
        camera = PiCamera()
        camera.rotation = 180
        camera.resolution = (resolution["width"], resolution["height"])
        camera.framerate = framerate
        rawCapture = PiRGBArray(camera, size = (resolution["width"], resolution["height"]))
        detections = self.detect()
        
        file = open("test_file", "w")
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        file.write("Active on: " + str(current_time))
        file.close()
        start_time = time.time()
        
        detection_results = None
        
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            img = frame.array    
            rawCapture.truncate(0)
            number_of_people = self.detector.detect_cv2(img)    
            current_time = time.time()
            elapsed_time = current_time - start_time
            self.database_manager.insertDetection(current_time, number_of_people, self.detection_period_id)
            if elapsed_time > self.detectionSeconds:
                break

        camera.stop_preview()
        camera.close()
        cv2.destroyAllWindows()