from subprocess import call
import sys
import os
from database_manager import DatabaseManager
import datetime

def schedule_detection(start_date, start_time, end_time, neural_network_type="SSD_Mobilenet_v2_320x320", obj_threshold=0.3, box_overlap_threshold=0.5,
                       video_resolution={"width":320, "height":320}, framerate=30):
    database_manager = DatabaseManager()

    detection_seconds = get_detection_seconds(start_time, end_time)

    detection_period_id = database_manager.insert_detection_period(time_date_to_timestamp(start_date, start_time), time_date_to_timestamp(start_date, end_time), neural_network_type, obj_threshold, video_resolution, framerate)
    command_string_list = []
    command_string_list.append("echo ")
    command_string_list.append("\"python3.7 ")
    command_string_list.append("/home/pi/Desktop/My_Server/backend/endpoints/scheduled_detection.py")
    command_string_list.append(" --detection_period_id=")
    command_string_list.append(str(detection_period_id))
    command_string_list.append(" --neural_network_type=")
    command_string_list.append(str(neural_network_type))
    command_string_list.append(" --detection_seconds=")
    command_string_list.append(str(detection_seconds))
    command_string_list.append(" --obj_threshold=")
    command_string_list.append(str(obj_threshold))
    command_string_list.append(" --box_overlap_threshold=")
    command_string_list.append(str(box_overlap_threshold))
    command_string_list.append(" --video_resolution_width=")
    command_string_list.append(str(video_resolution["width"]))
    command_string_list.append(" --video_resolution_height=")
    command_string_list.append(str(video_resolution["height"]))
    command_string_list.append(" --framerate=")
    command_string_list.append(str(framerate)+"\"")
    command_string_list.append(" | at ")
    command_string_list.append(time_dict_to_time_string(start_time))
    command_string_list.append(" ")
    command_string_list.append(date_dict_to_date_string(start_date))

    command = ''.join(command_string_list)
              
    print(command)
    
    os.system(command)
    
    
def get_detection_seconds(start_time, end_time):
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


def time_dict_to_time_string(time_dict):
    return time_dict["hour"] + ":" + time_dict["minute"]


def date_dict_to_date_string(date_dict):
    return date_dict["month"] + " " + date_dict["day"] + " " + date_dict["year"]

def time_date_to_timestamp(date_dict, time_dict):
    timezone_string = "+00:00"
    timestamp = datetime.datetime.strptime(date_dict["year"] + '-' + date_dict["month"] + '-' + date_dict["day"]
                                  + 'T' + time_dict["hour"] + ':' + time_dict["minute"] + timezone_string, '%Y-%b-%dT%H:%M%z')
    print("Timestamp: " + str(timestamp))
    return timestamp