import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from neural_networks_data import NEURAL_NETWORKS
from nms import non_max_suppression_fast

class Detector:
    def __init__(self, model_name):
        self.model = cv2.dnn.readNet(NEURAL_NETWORKS[model_name]["model"],
                        config=NEURAL_NETWORKS[model_name]["config"],
                        framework=NEURAL_NETWORKS[model_name]["framework"])
        self.class_names = NEURAL_NETWORKS[model_name]["class_names"]
        self.img_width = NEURAL_NETWORKS[model_name]["img_width"]
        self.img_height = NEURAL_NETWORKS[model_name]["img_height"]


    def detect(self, image):
        image_height, image_width, _ = image.shape
        # create blob from image
        blob = cv2.dnn.blobFromImage(image=image, size=(self.img_width, self.img_height), mean=(104, 117, 123), swapRB=True)
        # start time to calculate FPS
        start = time.time()
        self.model.setInput(blob)
        output = self.model.forward()       
        # end time after detection
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)
        boxes_with_people = []
        # loop over each of the detections
        for detection in output[0, 0, :, :]:
            # extract the confidence of the detection
            confidence = detection[2]
            # draw bounding boxes only if the detection confidence is above...
            # ... a certain threshold, else skip
            if confidence > .4:
                # get the class id
                class_id = detection[1]
                if class_id == 1:
                    # get the bounding box coordinates
                    box_x_1 = detection[3] * image_width
                    box_y_1 = detection[4] * image_height
                    # get the bounding box width and height
                    box_x_2 = detection[5] * image_width
                    box_y_2 = detection[6] * image_height
                    # draw a rectangle around each detected object
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (0, 255, 0), thickness=2)
                    # put the class name text on the detected object
                    cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # put the FPS text on top of the frame
                    cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    boxes_with_people.append([box_x_1, box_y_1, box_x_2, box_y_2])
        
        #uzycie algorytmu Non Maximal Suppresion
        filtered_boxes_with_people = non_max_suppression_fast(np.array(boxes_with_people))
        number_of_people = len(filtered_boxes_with_people)          
        cv2.imshow("Actual frame", img)
        cv2.waitKey(1) & 0xFF
        return number_of_people
