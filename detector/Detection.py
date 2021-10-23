######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
# 
# Modified by: Shawn Hymel
# Date: 09/22/20
# Description:
# Added ability to resize cv2 window and added center dot coordinates of each detected object.
# Objects and center coordinates are printed to console.

# Import packages

import sys
sys.path.append('/usr/local/lib/python3.7/dist-packages')
import os
import argparse
import cv2
import numpy as np
import time
from threading import Thread
import importlib.util
import tensorflow as tf
from yolo_functions import IMAGE_W, IMAGE_H, TRUE_BOX_BUFFER, ImageReader, ANCHORS, BoundBox, BestAnchorBoxFinder


class OutputRescaler(object):
    def __init__(self, ANCHORS):
        self.ANCHORS = ANCHORS

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def _softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)

        if np.min(x) < t:
            x = x / np.min(x) * t

        e_x = np.exp(x)
        return e_x / e_x.sum(axis, keepdims=True)

    def get_shifting_matrix(self, netout):

        GRID_H, GRID_W, BOX = netout.shape[:3]
        no = netout[..., 0]

        ANCHORSw = self.ANCHORS[::2]
        ANCHORSh = self.ANCHORS[1::2]

        mat_GRID_W = np.zeros_like(no)
        for igrid_w in range(GRID_W):
            mat_GRID_W[:, igrid_w, :] = igrid_w

        mat_GRID_H = np.zeros_like(no)
        for igrid_h in range(GRID_H):
            mat_GRID_H[igrid_h, :, :] = igrid_h

        mat_ANCHOR_W = np.zeros_like(no)
        for ianchor in range(BOX):
            mat_ANCHOR_W[:, :, ianchor] = ANCHORSw[ianchor]

        mat_ANCHOR_H = np.zeros_like(no)
        for ianchor in range(BOX):
            mat_ANCHOR_H[:, :, ianchor] = ANCHORSh[ianchor]
        return (mat_GRID_W, mat_GRID_H, mat_ANCHOR_W, mat_ANCHOR_H)

    def fit(self, netout):
        '''
        netout  : np.array of shape (N grid h, N grid w, N anchor, 4 + 1 + N class)

        a single image output of model.predict()
        '''
        GRID_H, GRID_W, BOX = netout.shape[:3]

        (mat_GRID_W,
         mat_GRID_H,
         mat_ANCHOR_W,
         mat_ANCHOR_H) = self.get_shifting_matrix(netout)

        # bounding box parameters
        netout[..., 0] = (self._sigmoid(netout[..., 0]) + mat_GRID_W) / GRID_W  # x      unit: range between 0 and 1
        netout[..., 1] = (self._sigmoid(netout[..., 1]) + mat_GRID_H) / GRID_H  # y      unit: range between 0 and 1
        netout[..., 2] = (np.exp(netout[..., 2]) * mat_ANCHOR_W) / GRID_W  # width  unit: range between 0 and 1
        netout[..., 3] = (np.exp(netout[..., 3]) * mat_ANCHOR_H) / GRID_H  # height unit: range between 0 and 1
        # rescale the confidence to range 0 and 1
        netout[..., 4] = self._sigmoid(netout[..., 4])
        expand_conf = np.expand_dims(netout[..., 4], -1)  # (N grid h , N grid w, N anchor , 1)
        # rescale the class probability to range between 0 and 1
        # Pr(object class = k) = Pr(object exists) * Pr(object class = k |object exists)
        #                      = Conf * P^c
        netout[..., 5:] = expand_conf * self._softmax(netout[..., 5:])
        # ignore the class probability if it is less than obj_threshold

        return (netout)
    
    
def find_high_class_probability_bbox(netout_scale, obj_threshold):
    '''
    == Input ==
    netout : y_pred[i] np.array of shape (GRID_H, GRID_W, BOX, 4 + 1 + N class)

             x, w must be a unit of image width
             y, h must be a unit of image height
             c must be in between 0 and 1
             p^c must be in between 0 and 1
    == Output ==

    boxes  : list containing bounding box with Pr(object is in class C) > 0 for at least in one class C


    '''
    GRID_H, GRID_W, BOX = netout_scale.shape[:3]

    boxes = []
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                # from 4th element onwards are confidence and class classes
                classes = netout_scale[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout_scale[row, col, b, :4]
                    confidence = netout_scale[row, col, b, 4]
                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes)
                    if box.get_score() > obj_threshold:
                        boxes.append(box)
    return (boxes)
 
 
def nonmax_suppression(boxes, iou_threshold, obj_threshold):
    '''
    boxes : list containing "good" BoundBox of a frame
            [BoundBox(),BoundBox(),...]
    '''
    bestAnchorBoxFinder = BestAnchorBoxFinder([])

    CLASS = len(boxes[0].classes)
    index_boxes = []
    # suppress non-maximal boxes
    for c in range(CLASS):
        # extract class probabilities of the c^th class from multiple bbox
        class_probability_from_bbxs = [box.classes[c] for box in boxes]

        # sorted_indices[i] contains the i^th largest class probabilities
        sorted_indices = list(reversed(np.argsort(class_probability_from_bbxs)))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            # if class probability is zero then ignore
            if boxes[index_i].classes[c] == 0:
                continue
            else:
                index_boxes.append(index_i)
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    # check if the selected i^th bounding box has high IOU with any of the remaining bbox
                    # if so, the remaining bbox' class probabilities are set to 0.
                    bbox_iou = bestAnchorBoxFinder.bbox_iou(boxes[index_i], boxes[index_j])
                    if bbox_iou >= iou_threshold:
                        classes = boxes[index_j].classes
                        classes[c] = 0
                        boxes[index_j].set_class(classes)

    newboxes = [boxes[i] for i in index_boxes if boxes[i].get_score() > obj_threshold]

    return newboxes




# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

def detect():
    IMAGE_W = 416
    IMAGE_H = 416
    TRUE_BOX_BUFFER = 50
    
    MODEL_2_TFLITE = "/home/pi/Desktop/PeopleCounting/RPIObjectDetection/TFLite/2021-09-02_2_/model.tflite"
    interpreter = tf.lite.Interpreter(MODEL_2_TFLITE)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(IMAGE_W,IMAGE_H),framerate=30).start()
    time.sleep(1)

    # Create window
    cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
    print('cv2.namedWindow(Object detector, cv2.WINDOW_NORMAL)')

    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    #while True:
    #for i in range(0, 1):

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (IMAGE_W, IMAGE_H))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    imageReader = ImageReader(IMAGE_H,IMAGE_W=IMAGE_W, norm=lambda image : image / 255.)
    print('imageReader = ImageReader(IMAGE_H,IMAGE_W=IMAGE_W, norm=lambda image : image / 255.)')
    out = imageReader.fit("/home/pi/Desktop/min.jpg")
    #out = imageReader.fit(train_image_folder + "/2007_005430.jpg")
    print(out.shape)
    X_test = np.expand_dims(out,0).astype('float32')
    print(X_test.shape)
    # handle the hack input
    dummy_array = np.ones((1,1,1,1,TRUE_BOX_BUFFER,4)).astype('float32')
    print(dummy_array.dtype)
    interpreter.set_tensor(input_details[0]['index'], dummy_array)
    interpreter.set_tensor(input_details[1]['index'], X_test)
    interpreter.invoke()
    y_pred = interpreter.get_tensor(output_details[0]['index'])
    print('y_pred = interpreter.get_tensor(output_details[0][index])')
    
    netout         = y_pred[0]
    outputRescaler = OutputRescaler(ANCHORS=ANCHORS)
    netout_scale   = outputRescaler.fit(netout)
    
    obj_threshold = 0.03
    boxes = find_high_class_probability_bbox(netout_scale,obj_threshold)
    
    iou_threshold = 0.1
    final_boxes = nonmax_suppression(boxes,iou_threshold=iou_threshold,obj_threshold=obj_threshold)
    print('final_boxes = nonmax_suppression(boxes,iou_threshold=iou_threshold,obj_threshold=obj_threshold)')
    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
    
    return final_boxes[0]