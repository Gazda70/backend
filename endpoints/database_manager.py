import pymongo
import datetime
import math

class DatabaseManager:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["PeopleCounterDatabase"]
        self.detection_period_collection = self.db["DetectionPeriod"]
        self.detection_collection = self.db["Detection"]
        
        
    def insert_detection_period(self, start_time, end_time, neural_network_type, detection_seconds,  obj_threshold,
                       video_resolution, framerate):  
        return_value = self.detection_period_collection.insert_one({"start_time":start_time, "end_time":end_time, "detection_seconds":detection_seconds,
                                                          "neural_network_type":neural_network_type, "obj_threshold":obj_threshold, "video_resolution":video_resolution, "framerate":framerate})
        return return_value.inserted_id
    
        
    def insert_detection(self, time, detections, detection_period_id):
        self.detection_collection.insert_one({"time":time, "detections":detections, "detection_period_id":str(detection_period_id)})
        
        
    def find_detection_periods_for_given_date(self, date):
        next_day_date = date + datetime.timedelta(days=1)
        criteria = {"start_time":{"$gte":date, "$lt": next_day_date}}
        return self.detection_period_collection.find(criteria)
    
    
    def find_all_detections_for_given_detection_period(self, detection_period_id):
        criteria = {"detection_period_id":detection_period_id}
        return self.detection_collection.find(criteria)
        