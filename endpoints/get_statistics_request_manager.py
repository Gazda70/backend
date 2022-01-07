import json
import time
import math
import ast
from database_manager import DatabaseManager
from numpy.lib.function_base import average
import datetime

class GetStatisticsRequestManager:
    def __init__(self):
        self.database_manager = DatabaseManager()
    
    def manage_request(self, req):
        detection_period_list = self.get_detection_periods(req)
        detection_period_stats = []
        total_averaged_detections = []
        for det_per in detection_period_list:
            detections = self.database_manager.find_all_detections_for_given_detection_period(str(det_per.get('_id')))
            detections_list = list(detections)
            print("Detection period")
            print(det_per)
            print("Detections")
            print(detections_list)
            averaged_detections = self.divide_into_subcollections(detections_list, 2)
            total_averaged_detections = total_averaged_detections + averaged_detections
            if len(averaged_detections) != 0:
                print("len(averaged_detections): " + str(len(averaged_detections)))
                print("average(averaged_detections): " + str(average(averaged_detections)))
                people_min = min(averaged_detections)
                people_max = max(averaged_detections)
                people_avg = math.ceil(average(averaged_detections))
            else:
                people_min = 0
                people_max = 0
                people_avg = 0
            
            start_time = det_per["start_time"].strftime("%m%d%Y %H:%M").split(' ')[1]
            end_time = det_per["end_time"].strftime("%m%d%Y %H:%M").split(' ')[1]
            detection_period_stats.append({"start_time":start_time, "end_time":end_time, "people_min":people_min,
                                           'people_max':people_max, 'people_avg':people_avg})
            
        if len(total_averaged_detections) != 0: 
            people_min = min(total_averaged_detections)
            people_max = max(total_averaged_detections)
            people_avg = math.ceil(average(total_averaged_detections))
        else:
            people_min = 0
            people_max = 0
            people_avg = 0   
        whole_day_stats = {"people_min":people_min,
                           'people_max':people_max,
                           'people_avg':people_avg}
        response_body = {"detection_period_stats":detection_period_stats, "whole_day_stats":whole_day_stats}
        print(response_body)
        return response_body
    
    
    def get_detection_periods(self, req):
        start_date = self.date_to_timestamp(ast.literal_eval(req["startDate"]))
        end_date = start_date + datetime.timedelta(days=1)
        results = self.database_manager.find_detection_periods_for_given_date(self.date_to_timestamp(ast.literal_eval(req["startDate"])))
        return list(results)
        

    def time_date_to_timestamp(self, time_dict, date_dict):
        timezone_string = "+00:00"
        timestamp = datetime.datetime.strptime(date_dict["year"] + '-' + date_dict["month"] + '-' + date_dict["day"]
                                      + 'T' + time_dict["hour"] + ':' + time_dict["minute"] + timezone_string, '%Y-%b-%dT%H:%M%z')
        return timestamp


    def date_to_timestamp(self, date_dict):
        timezone_string = "+00:00"
        timestamp = datetime.datetime.strptime(date_dict["year"] + '-' + date_dict["month"] + '-' + date_dict["day"],'%Y-%b-%d')
        return timestamp
    
    def divide_into_subcollections(self, detections, border_frame_number):
        averaged_detections = []
        count_frame_number = 1
        summed_people_number = 0
        for detection in detections:
            summed_people_number = summed_people_number + detection["detections"]
            if count_frame_number >= border_frame_number:
                averaged_detections.append(int(summed_people_number/count_frame_number))
                count_frame_number = 1
                summed_people_number = 0
            else:
                count_frame_number = count_frame_number + 1
        if count_frame_number != 0:
            averaged_detections.append(int(summed_people_number/count_frame_number))
        return averaged_detections
