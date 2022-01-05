import sys
sys.path.append("/usr/lib/python3.7/")
import platform
import cv2
from datetime import datetime
import datetime
import time
import numpy as np
import os
#from detector import Detector
from picamera.array import PiRGBArray
from picamera import PiCamera
#from database_manager import DatabaseManager
import ast
import tensorflow as tf
#from neural_networks_data import NEURAL_NETWORKS
import pymongo


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
            '''
            file = open("test_file", "a")
            file.write("Elapsed time: " + str(current_time))
            file.close()
            '''
            if elapsed_time > self.detection_seconds:
                break

        camera.stop_preview()
        camera.close()
        cv2.destroyAllWindows()
        
        
class DatabaseManager:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["PeopleCounterDatabase"]
        self.detection_period_collection = self.db["DetectionPeriod"]
        self.detection_collection = self.db["Detection"]
        
        
    def insert_detection_period(self, start_time, end_time, neural_network_type, obj_threshold, video_resolution, framerate):  
        return_value = self.detection_period_collection.insert_one({"start_time":start_time, "end_time":end_time,
                                                          "neural_network_type":neural_network_type, "obj_threshold":obj_threshold, "video_resolution":video_resolution, "framerate":framerate})
        return return_value.inserted_id
    
        
    def insert_detection(self, time, detections, detection_period_id):
        self.detection_collection.insert_one({"time":time, "detections":detections, "detection_period_id":str(detection_period_id)})
        
        
    def find_detection_periods_for_given_date(self, date):
        next_day_date = date + datetime.timedelta(days=1)
        criteria = {"start_time":{"$gte":date, "$lt": next_day_date}}
        return self.detection_period_collection.find(criteria)
    
    
    def find_all_detections_for_given_detection_period(self, detection_period_id):
        criteria = {"detection_period_id":detection_period_id}
        return self.detection_collection.find(criteria)
    

class Detector:
    def __init__(self, model_name, object_threshold, box_overlap_threshold):
        self.model = cv2.dnn.readNet(NEURAL_NETWORKS[model_name]["model"],
                        config=NEURAL_NETWORKS[model_name]["config"],
                        framework=NEURAL_NETWORKS[model_name]["framework"])
        self.img_width = NEURAL_NETWORKS[model_name]["img_width"]
        self.img_height = NEURAL_NETWORKS[model_name]["img_height"]
        self.object_threshold = object_threshold
        self.box_overlap_threshold = box_overlap_threshold

    def detect(self, image):
        image_height, image_width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image=image, size=(self.img_width, self.img_height), mean=(104, 117, 123), swapRB=True)
        start = time.time()
        self.model.setInput(blob)
        output = self.model.forward()
        end = time.time()
        fps = 1 / (end-start)
        boxes_with_people = []
        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > self.object_threshold:
                class_id = detection[1]
                if class_id == 1:
                    box_x_1 = detection[3] * image_width
                    box_y_1 = detection[4] * image_height
                    box_x_2 = detection[5] * image_width
                    box_y_2 = detection[6] * image_height
                    #cv2.rectangle(image, (int(box_x_1), int(box_y_1)), (int(box_x_2), int(box_y_2)), (0, 0, 255), thickness=2)
                    #cv2.putText(image, "TARGET ACQUIRED", (int(box_x_1), int(box_y_2 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #cv2.putText(image, "SSD Mobilenet v2", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    boxes_with_people.append([box_x_1, box_y_1, box_x_2, box_y_2])
        
        filtered_boxes_with_people = non_max_suppression_fast(np.array(boxes_with_people), self.box_overlap_threshold)
        #filtered_boxes_with_people = boxes_with_people
        number_of_people = len(filtered_boxes_with_people)
        file = open("test_file", "a")
        file.write("Number of people: " + str(number_of_people))
        file.close()
        #cv2.imshow("Actual frame", image)
        #cv2.waitKey(1) & 0xFF
        return number_of_people
    
def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
    return boxes[pick]

NEURAL_NETWORKS = {"SSD_Mobilenet_v2_320x320":{"model":"/home/pi/Desktop/My_Server/backend/models/frozen_inference_graph.pb",
                    "config":"/home/pi/Desktop/My_Server/backend/models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt",
                                               "img_width":320, "img_height":320, "class_names":[
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush'
], "framework":"OpenCV"}}
