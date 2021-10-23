import cv2
from threading import Thread
from video_stream import VideoStream
from datetime import datetime
import datetime
import time
import tensorflow as tf
import numpy as np
import os
from yolo_functions import OutputRescaler, ImageReader, find_high_class_probability_bbox, nonmax_suppression, ANCHORS, TRUE_BOX_BUFFER, LABELS


class DetectionManager:
    def __init__(self):
        self.PATH_TO_DETECTION_FILES = "/home/pi/Desktop/Backend/PortableHumanRecognitionSystemWebApplication/backend/DetectionData/"
        self.SSD_INFERENCE_GRAPH = '/home/pi/Desktop/PeopleCounting/RPIObjectDetection/Code/Detection/SSD/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
        self.SSD_PBTXT = '/home/pi/Desktop/PeopleCounting/RPIObjectDetection/Code/Detection/SSD/ssd_mobilenet_v2_coco_2018_03_29.pbtxt'

    def startDetection(self, neuralNetworkType, detectionSeconds,  obj_threshold=0.3, iou_threshold = 0.1):
        self.neuralNetworkType = neuralNetworkType
        self.obj_threshold = obj_threshold
        self.iou_threshold = iou_threshold
        self.detectionSeconds = detectionSeconds
        print("Starting detection: " + str(datetime.datetime.now()))
        if self.neuralNetworkType == "CUSTOM":
            netout = self.load_model_GazdaWitekLipka()
        elif self.neuralNetworkType == "SSD":
            netout = self.load_model_SSD()
        else:
            netout=None
        detections = self.detect()
        print("Ending detection: " + str(datetime.datetime.now()))

    def load_model_SSD(self):
        self.model_SSD = cv2.dnn.readNetFromTensorflow(
            self.SSD_INFERENCE_GRAPH,
            self.SSD_PBTXT)

    def detect_SSD(self, image, img_w, img_h):
        frame = image.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_w, img_h))
        self.model_SSD.setInput(cv2.dnn.blobFromImage(frame_resized, size=(img_w, img_h), swapRB=True))
        output = self.model_SSD.forward()
        return output

    def load_model_GazdaWitekLipka(self):
        MODEL_2_TFLITE = "/home/pi/Desktop/PeopleCounting/RPIObjectDetection/TFLite/2021-09-02_2_/model.tflite"
        self.model_GazdaWitekLipka = tf.lite.Interpreter(MODEL_2_TFLITE)
        self.model_GazdaWitekLipka.allocate_tensors()

    def detect_GazdaWitekLipka(self, image, img_w, img_h):
        frame = image.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_w, img_h))
        input_details = self.model_GazdaWitekLipka.get_input_details()
        output_details = self.model_GazdaWitekLipka.get_output_details()
        #imageReader = ImageReader(img_h, IMAGE_W=img_w, norm=lambda image: image / 255.)
        #out = imageReader.fit(frame_resized)
        X_test = np.expand_dims(frame_resized, 0).astype('float32')
        dummy_array = np.ones((1, 1, 1, 1, TRUE_BOX_BUFFER, 4)).astype('float32')
        self.model_GazdaWitekLipka.set_tensor(input_details[0]['index'], dummy_array)
        self.model_GazdaWitekLipka.set_tensor(input_details[1]['index'], X_test)
        self.model_GazdaWitekLipka.invoke()
        y_pred = self.model_GazdaWitekLipka.get_tensor(output_details[0]['index'])

        return y_pred


    def adjust_minmax(self, c, _max):
        if c < 0:
            c = 0
        if c > _max:
            c = _max
        return c


    def detect(self):
        IMAGE_W = 416
        IMAGE_H = 416

        detection_objects = []
        # Initialize video stream
        videostream = VideoStream(resolution=(IMAGE_W, IMAGE_H), framerate=30).start()
        time.sleep(1)

        # Create window
        cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)

        frame = videostream.read()
        
        

        start_time = time.time()
        while True:
            if self.neuralNetworkType == "CUSTOM":
                y_pred = self.detect_GazdaWitekLipka(frame, IMAGE_W, IMAGE_H)
            elif self.neuralNetworkType == "SSD":
                y_pred = self.detect_SSD(frame, 300, 300)
            else:
                y_pred=None
                
            netout = y_pred[0]

            outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
            netout_scale = outputRescaler.fit(netout)

            boxes = find_high_class_probability_bbox(netout_scale, self.obj_threshold)

            iou_threshold = 0.1
            final_boxes = nonmax_suppression(boxes, iou_threshold=iou_threshold, obj_threshold=self.obj_threshold)

            obj_baseline = 0.05

            score_rescaled = np.array([box.get_score() for box in final_boxes])
            score_rescaled /= obj_baseline

            predicted_boxes = []

            for sr, box in zip(score_rescaled, boxes):
                print('PREDICTED LABEL: ' + LABELS[box.label])
                xmin = self.adjust_minmax(int(box.xmin * IMAGE_W), IMAGE_W)
                ymin = self.adjust_minmax(int(box.ymin * IMAGE_H), IMAGE_H)
                xmax = self.adjust_minmax(int(box.xmax * IMAGE_W), IMAGE_W)
                ymax = self.adjust_minmax(int(box.ymax * IMAGE_H), IMAGE_H)
                if LABELS[box.label] == 'person':
                    predicted_boxes.append(
                        {'name': LABELS[box.label], 'score': box.get_score(),
                                                        'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

            current_time = time.time()
            elapsed_time = current_time - start_time
            detection_objects.append({"frame_time":current_time, "detections":predicted_boxes})
            if elapsed_time > self.detectionSeconds:
                self.writeDetectionPeriodSummary(detection_objects, start_time, self.detectionSeconds)
                '''
                for det_obj in detection_objects:
                    for fin_box in det_obj:
                        print("detection_object: " + str(fin_box.classes[0]))
                '''
                break

        cv2.destroyAllWindows()
        videostream.stop()

        return final_boxes


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