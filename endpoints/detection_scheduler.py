from subprocess import call
import sys
import os
from database_manager import DatabaseManager

def schedule_detection(start_date, start_time, end_time, neuralNetworkType="SSD Mobilenet v2 320x320", detectionSeconds=10,  obj_threshold=0.3,
                       video_resolution={"width":320, "height":320}, framerate=30):
    print("Start date: " + start_date)
    print("Start time: " + start_time)
    
    database_manager = DatabaseManager()
    
    detection_period_id = database_manager.insertDetectionPeriod(start_date, start_time, end_time, neuralNetworkType, detectionSeconds,  obj_threshold,
                       video_resolution, framerate)
    
    os.system("python " + "scheduled_detection.py" + " --neuralNetworkType=" + str(neuralNetworkType) + " --detectionSeconds=" + str(detectionSeconds) + " --obj_threshold=" + str(obj_threshold) +
                       " --video_resolution_width=" + str(video_resolution["width"]) + " --video_resolution_height=" + str(video_resolution["height"]) +
              " --framerate=" + str(framerate) + " --detection_period_id=" + str(detection_period_id) + 
    " | at " + str(start_time) + " " + str(start_date))