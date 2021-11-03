import cv2
from threading import Thread
from video_stream import VideoStream
from datetime import datetime
import datetime
import time
import tensorflow as tf
import numpy as np
import os
from detector import Detector
from picamera.array import PiRGBArray
from picamera import PiCamera
from database_manager import DatabaseManager
import pymongo

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
image = cv2.imread("/home/pi/Desktop/happyPeople.jpg")


class DetectionManager:
    def __init__(self):
        self.model_paths = {"SSD_Mobilenet_v2_320x320":"/home/pi/Desktop/My_Server/backend/models/"}
        self.category_maps = {"SSD_Mobilenet_v2_320x320":{1: "person"}}

    def setupDetection(self, detection_period_id, neuralNetworkType="SSD_Mobilenet_v2_320x320", detectionSeconds=10,  obj_threshold=0.3,
                       video_resolution={"width":320, "height":320}, framerate=30):
        self.neuralNetworkType = neuralNetworkType
        self.obj_threshold = obj_threshold
        self.detectionSeconds = detectionSeconds
        self.video_resolution = video_resolution
        self.framerate = framerate
        self.detector = Detector(self.model_paths["SSD_Mobilenet_v2_320x320"], self.category_maps["SSD_Mobilenet_v2_320x320"])
        self.database_manager = DatabaseManager()
        self.detection_period_id = detection_period_id
        print("Starting detection: " + str(datetime.datetime.now()))

        detections = self.detect()
        print("Ending detection: " + str(datetime.datetime.now()))

    def detect(self):
        resolution={"width":320, "height":320}
        framerate=30
        camera = PiCamera()
        camera.resolution = (resolution["width"], resolution["height"])
        camera.framerate = framerate
        rawCapture = PiRGBArray(camera, size = (resolution["width"], resolution["height"]))
        
        start_time = time.time()
        
        #video_stream = VideoStream(self.video_resolution, self.framerate)
        
        #video_stream.start()
        
    
            #frame = video_stream.read()
        for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

            # Otherwise, grab the next frame from the stream
            frame = frame.array
            
            rawCapture.truncate(0)

            detection_results = self.detector.detect(frame, self.video_resolution["width"], self.video_resolution["height"])

            current_time = time.time()
            elapsed_time = current_time - start_time
            
            #detection_objects.append({"frame_time":current_time, "detections":predicted_boxes})
            self.database_manager.insertDetection(current_time, detection_results, self.detection_period_id)
            
            if elapsed_time > self.detectionSeconds:
                #self.writeDetectionPeriodSummary(detection_objects, start_time, self.detectionSeconds)
                '''
                for det_obj in detection_objects:
                    for fin_box in det_obj:
                        print("detection_object: " + str(fin_box.classes[0]))
                '''
                break

        camera.stop_preview()
        camera.close()
        cv2.destroyAllWindows()
        #videostream.stop()

        #return final_boxes


    def determineSecondsForDetection(self, detectionTimeString):
        timeValues = []
        timeValues = detectionTimeString.split(':')
        for timeVal in timeValues:
            print("Time value: " + timeVal)
        timeNow = datetime.datetime.now().time()
        hours = int(timeValues[0]) - timeNow.hour
        minutes = int(timeValues[1]) - timeNow.minute
        print("Hours: " + str(hours))
        print("Minutes: " + str(minutes))
        if minutes < 0:
            minutes = 60 - minutes
            hours -= 1

        if hours < 0:
            print("End time must be grater that start time !")
        start_time = time.time()


    def writeDetectionPeriodSummary(self, detections, timestamp, seconds_of_detection):
        f = open(self.PATH_TO_DETECTION_FILES + str(timestamp), "a")
        f.write("START:" + str(timestamp) + ":DURATION:" + str(seconds_of_detection) + "\n")
        for detection in detections:
            f.write("frame_time:" + str(detection["frame_time"]) + ":detections:" + str(detection["detections"]) + "\n")
        f.write("NUMBER_OF_DETECTIONS:" + str(len(detections)))
        f.close()
        
        
    def get_detection_data(self):
        detection_objects = []
        # iterate over files in
        # that directory
        for filename in os.listdir(self.PATH_TO_DETECTION_FILES):
            f = os.path.join(self.PATH_TO_DETECTION_FILES, filename)
            # checking if it is a file
            if os.path.isfile(f):
                myfile = open(f, "r")
                mylist = myfile.readlines()
                basic_info_array = mylist[0].split(':')
                print("Timestamp: " + basic_info_array[1])
                print("secondsOfDetection: " + basic_info_array[3])
                detection_object = {
                    "timestamp":basic_info_array[1],
                    "secondsOfDetection":basic_info_array[3],
                    "detections":mylist[1:len(mylist)-1],
                    "numberOfDetections":mylist[-1]
                    }
                myfile.close()
                return detection_object
                '''
                detection_objects.append(detection_object)
        return detection_objects
'''