from subprocess import call
import sys
import os
from database_manager import DatabaseManager
import datetime

def schedule_detection(start_date, start_time, end_time, neuralNetworkType="SSD_Mobilenet_v2_320x320", obj_threshold=0.3,
                       video_resolution={"width":320, "height":320}, framerate=30):
    #print("Start date: " + start_date)
    #print("Start time: " + start_time)
    database_manager = DatabaseManager()
    
    detectionSeconds = getDetectionSeconds(start_time, end_time)
    
    print("detectionSeconds: \n")
    print(detectionSeconds)
    '''
    detection_period_id = database_manager.insertDetectionPeriod(dateDictToDateString(start_date), timeDictToTimeString(start_time),
                                                                 timeDictToTimeString(end_time), neuralNetworkType, detectionSeconds,  obj_threshold,
                       video_resolution, framerate)
    '''
    detection_period_id = database_manager.insertDetectionPeriod(time_date_to_timestamp(start_date, start_time), time_date_to_timestamp(start_date, end_time),
                                                                 neuralNetworkType, detectionSeconds, obj_threshold, video_resolution, framerate)
    command_string_list = []
    command_string_list.append("python3.7 ")
    command_string_list.append("scheduled_detection.py")
    command_string_list.append(" --detection_period_id=")
    command_string_list.append(str(detection_period_id))
    command_string_list.append(" --neuralNetworkType=")
    command_string_list.append(str(neuralNetworkType))
    command_string_list.append(" --detectionSeconds=")
    command_string_list.append(str(detectionSeconds))
    command_string_list.append(" --obj_threshold=")
    command_string_list.append(str(obj_threshold))
    command_string_list.append(" --video_resolution_width=")
    command_string_list.append(str(video_resolution["width"]))
    command_string_list.append(" --video_resolution_height=")
    command_string_list.append(str(video_resolution["height"]))
    command_string_list.append(" --framerate=")
    command_string_list.append(str(framerate))
    command_string_list.append(" | at ")
    command_string_list.append(timeDictToTimeString(start_time))
    command_string_list.append(" ")
    command_string_list.append(dateDictToDateString(start_date))
    
    command = ''.join(command_string_list)
              
    print(command)
    
    os.system(command)
    
    
def getDetectionSeconds(start_time, end_time):
    start_hour = int(start_time["hour"])
    start_minute = int(start_time["minute"])
    
    end_hour = int(end_time["hour"])
    end_minute = int(end_time["minute"])
    
    result_minute = 0
    
    if start_hour <= end_hour:
        hour = end_hour - start_hour
        minute = end_minute - start_minute
        result_minute = hour * 60 + minute
        print("result_minute: ")
        print(result_minute)
    
    return result_minute * 60


def timeDictToTimeString(time_dict):
    return time_dict["hour"] + ":" + time_dict["minute"]


def dateDictToDateString(date_dict):
    return date_dict["month"] + " " + date_dict["day"] + " " + date_dict["year"]

def time_date_to_timestamp(date_dict, time_dict):
    timezone_string = "+00:00"
    timestamp = datetime.datetime.strptime(date_dict["year"] + '-' + date_dict["month"] + '-' + date_dict["day"]
                                  + 'T' + time_dict["hour"] + ':' + time_dict["minute"] + timezone_string, '%Y-%b-%dT%H:%M%z')
    print("Timestamp: " + str(timestamp))
    return timestamp