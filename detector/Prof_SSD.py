import cv2
'''
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
'''
from os.path import exists
import numpy as np
import tensorflow as tf
import copy
import os
import xml.etree.ElementTree as ET
import sys
import tensorflow.keras.backend as K
from tensorflow import keras
from threading import Thread

import time

PATH_TO_DETECTION_FILES = ""

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

def detectSSD(numberOfSeconds, timestamp):
    model = cv2.dnn.readNetFromTensorflow(
        '/home/pi/Desktop/PeopleCounting/RPIObjectDetection/Code/Detection/SSD/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
        '/home/pi/Desktop/PeopleCounting/RPIObjectDetection/Code/Detection/SSD/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(300, 300),framerate=30).start()
    #time.sleep(1)

    ########### MEASUREMENT CODE #################
    NUMBER_OF_DETECTIONS = 0
    RESULT_FILE_PATH="/home/pi/Desktop/PeopleCounting/RPIObjectDetection/Code/Detection/results_ssd_2.txt"
    frame_rate_table = []
    start = time.time()

    ########### MEASUREMENT CODE #################

    # Create window
    #cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)
    #print("Reached here")
    #for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    while True:
    #for i in range(0, 10):
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()
        cv2.imshow('Object detector', frame1)
        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (300, 300))

        model.setInput(cv2.dnn.blobFromImage(frame_resized, size=(300, 300), swapRB=True))
        output = model.forward()
        # Loop over all detections and draw detection box if confidence is above minimum threshold


        ########### MEASUREMENT CODE #################

        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        frame_rate_table.append(frame_rate_calc)

        #Calculate total time
        NUMBER_OF_DETECTIONS += 1
        end = time.time()
        elapsed_time = end-start
        if elapsed_time >= numberOfSeconds:
            writeDetectionPeriodSummary(timestamp)
            break

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()


def writeDetectionPeriodSummary(timestamp):
    f = open(PATH_TO_DETECTION_FILES + timestamp, "a")
    f.write("Framerates:\n")
    for i in frame_rate_table:
        f.write(str(i) + "\n")
    f.write("NUMBER_OF_DETECTIONS: " + str(NUMBER_OF_DETECTIONS))
    f.close()
