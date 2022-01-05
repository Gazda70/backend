import sys
sys.path.append("/usr/lib/python3.7/")
import platform
import cv2
from datetime import datetime
import datetime
import time
import numpy as np
import os
from detector import Detector
from picamera.array import PiRGBArray
from picamera import PiCamera
from database_manager import DatabaseManager
import ast
import tensorflow as tf
from neural_networks_data import NEURAL_NETWORKS


class DetectionManager:

    def setupDetection(self, detection_period_id, neural_network_type, detection_seconds,
                       obj_threshold, box_overlap_threshold, framerate, video_resolution):
        self.neural_network_type = neural_network_type
        self.obj_threshold = obj_threshold
        self.detection_seconds = detection_seconds
        self.framerate = framerate
        self.detector = Detector(neural_network_type, obj_threshold, box_overlap_threshold)
        self.database_manager = DatabaseManager()
        self.detection_period_id = detection_period_id
        self.video_resolution = video_resolution
        self.box_overlap_threshold = box_overlap_threshold
        self.detection_seconds = detection_seconds
        detections = self.detect()

    def detect(self):
        framerate=30
        camera = PiCamera()
        camera.rotation = 180
        camera.resolution = (self.video_resolution["width"], self.video_resolution["height"])
        camera.framerate = framerate
        rawCapture = PiRGBArray(camera, size = (self.video_resolution["width"], self.video_resolution["height"]))
        start_time = time.time()

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            img = frame.array    
            rawCapture.truncate(0)
            number_of_people = self.detector.detect(img)    
            current_time = time.time()
            elapsed_time = current_time - start_time
            self.database_manager.insert_detection(current_time, number_of_people, self.detection_period_id)
            if elapsed_time > self.detection_seconds:
                break

        camera.stop_preview()
        camera.close()
        cv2.destroyAllWindows()