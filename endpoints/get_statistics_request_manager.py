import json
import time
import math
import ast
from detection_scheduler import schedule_detection
from database_manager import DatabaseManager
from statistics_calculator import StatisticsCalculator
import datetime

class GetStatisticsRequestManager:
    def __init__(self):
        self.database_manager = DatabaseManager()
    
    def manage_request(self, req):
        detection_period_list = self.get_detection_periods(req)
        detection_period_stats = []
        total_averaged_detections = []
        statistics_calculator = StatisticsCalculator()
        for det_per in detection_period_list:
            detections = self.database_manager.find_all_detections_for_given_detection_period(str(det_per.get('_id')))
            detections_list = list(detections)
            
            averaged_detections = statistics_calculator.divide_into_same_people_count_collections(detections_list, 2)
            total_averaged_detections = total_averaged_detections + averaged_detections
            people_min = statistics_calculator.min_detected_people(averaged_detections)
            people_max = statistics_calculator.max_detected_people(averaged_detections)
            people_avg = statistics_calculator.arithmetic_average_detected_people(averaged_detections)
            
            start_time = det_per["start_time"].strftime("%m%d%Y %H:%M:%S").split(' ')[1]
            end_time = det_per["end_time"].strftime("%m%d%Y %H:%M:%S").split(' ')[1]
            detection_period_stats.append({"start_time":start_time, "end_time":end_time, "people_min":people_min,
                                           'people_max':people_max, 'people_avg':people_avg})
            
            
        whole_day_stats = {"people_min":statistics_calculator.min_detected_people(total_averaged_detections),
                           'people_max':statistics_calculator.max_detected_people(total_averaged_detections),
                           'people_avg':statistics_calculator.arithmetic_average_detected_people(total_averaged_detections)}
        response_body = {"detection_period_stats":detection_period_stats, "whole_day_stats":whole_day_stats}
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