import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from neural_networks_data import NEURAL_NETWORKS
from nms import non_max_suppression_fast
from datetime import datetime

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
                    cv2.rectangle(image, (int(box_x_1), int(box_y_1)), (int(box_x_2), int(box_y_2)), (0, 0, 255), thickness=2)
                    cv2.putText(image, "TARGET ACQUIRED", (int(box_x_1), int(box_y_2 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    #cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, "SSD Mobilenet v2", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    boxes_with_people.append([box_x_1, box_y_1, box_x_2, box_y_2])
        
        file = open("test_file", "a")
        file.write("Detection starting")
        file.close()
        filtered_boxes_with_people = non_max_suppression_fast(np.array(boxes_with_people), self.box_overlap_threshold)
        number_of_people = len(filtered_boxes_with_people)
        cv2.imshow("Actual frame", image)
        cv2.waitKey(1) & 0xFF
        return number_of_people
