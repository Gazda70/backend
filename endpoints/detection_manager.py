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

    def setupDetection(self, detection_period_id, neural_network_type="SSD_Mobilenet_v2_320x320", detection_seconds=60,
                       object_threshold=0.3, box_overlap_threshold=0.8, framerate=30, video_resolution={"width":1260, "height":720}):
        self.neural_network_type = neural_network_type
        self.obj_threshold = obj_threshold
        self.detection_seconds = detection_seconds
        self.framerate = framerate
        self.detector = Detector(neural_network_type, object_threshold, box_overlap_threshold)
        self.database_manager = DatabaseManager()
        self.detection_period_id = detection_period_id
        self.video_resolution = video_resolution
        detections = self.detect()

    def detect(self):
        framerate=30
        camera = PiCamera()
        camera.rotation = 180
        camera.resolution = (self.video_resolution["width"], self.video_resolution["height"])
        camera.framerate = framerate
        rawCapture = PiRGBArray(camera, size = (self.video_resolution["width"], self.video_resolution["height"]))

        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            img = frame.array    
            rawCapture.truncate(0)
            number_of_people = self.detector.detect(img)    
            current_time = time.time()
            elapsed_time = current_time - start_time
            self.database_manager.insert_detection(current_time, number_of_people, self.detection_period_id)
            if elapsed_time > self.detectionSeconds:
                break

        camera.stop_preview()
        camera.close()
        cv2.destroyAllWindows()