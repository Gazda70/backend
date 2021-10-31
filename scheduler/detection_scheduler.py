from subprocess import call
import sys
import os
data = sys.argv[1]

def schedule_detection(start_date, start_time, neuralNetworkType="SSD Mobilenet v2 320x320", detectionSeconds=10,  obj_threshold=0.3,
                       video_resolution={"width":320, "height":320}, framerate=30):
    print("Start date: " + start_date)
    print("Start time: " + start_time)
    os.system("python " + 
    " | at " + start_time + " " + start_date)
    

schedule_detection(sys.argv[1], sys.argv[2])