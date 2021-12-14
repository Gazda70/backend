import pymongo
import datetime
import math

class DatabaseManager:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")

        self.db = self.client["PeopleCounterDatabase"]

        self.detection_period_collection = self.db["DetectionPeriod"]
        
        self.detection_collection = self.db["Detection"]
        
    def insertDetectionPeriod(self, start_time, end_time, neural_network_type, detection_seconds,  obj_threshold,
                       video_resolution, framerate):
        
        return_value = self.detection_period_collection.insert_one({"start_time":start_time, "end_time":end_time, "detection_seconds":detection_seconds,
                                                          "neural_network_type":neural_network_type, "obj_threshold":obj_threshold, "video_resolution":video_resolution, "framerate":framerate})
        '''
        return_value = self.detection_period_collection.insert_one({"start_date":start_date, "start_time":start_time, "end_time":end_time, "detection_seconds":detection_seconds,
                                                                  "neural_network_type":neural_network_type, "obj_threshold":obj_threshold,
                       "video_resolution":video_resolution, "framerate":framerate})
        '''
        return return_value.inserted_id
    
        
    def insertDetection(self, time, detections, detection_period_id):
        print("When inserting, detection_period_id: " + str(detection_period_id))
        self.detection_collection.insert_one({"time":time, "detections":detections, "detection_period_id":str(detection_period_id)})
        
        
    def findDetectionsForDetectionPeriod(self, detection_period_id):
        return self.detection_collection.find({"detection_period_id":detection_period_id})
        
        
    def findDetectionPeriodsForGivenDate(self, date):
        next_day_date = date + datetime.timedelta(days=1)
        criteria = {"start_time":{"$gte":date, "$lt": next_day_date}}
        return self.detection_period_collection.find(criteria)
    
    
    def findDetectionPeriodsForGivenDateRange(self, start_date_range, end_date_range):
        
        criteria = {"start_time":{"$gte":start_date_range, "$lte": end_date_range}}
        return self.detection_period_collection.find(criteria)
    
    def findAllDetectionsForGivenDetectionPeriod(self, detection_period_id):
        
        criteria = {"detection_period_id":detection_period_id}
        return self.detection_collection.find(criteria)
                
    def minDetectedPeople(self, detections):
       minimal = self.maxDetectedPeople(detections)
       for det in detections:
           if det["detections"] < minimal:
               minimal = det["detections"]
       return minimal
    
    def maxDetectedPeople(self, detections):
       maximal = 0
       for det in detections:
           if det["detections"] > maximal:
               maximal = det["detections"]
       return maximal
    
    def arithmeticAverageDetectedPeople(self, detections):
        people_count = 0
        for det in detections:
            people_count += det["detections"]
            '''
        print("Before int average: " + str(people_count/len(detections)))
        print("Average:"
              + str(math.ceil(people_count/len(detections))))
              '''
        #use math.ceil to get bigger average
        if len(detections) != 0:
            return math.ceil(people_count/len(detections))
        return 0
        
def divideIntoSamePeopleCountCollections(detections, minimal_same_count_frame_number):
    averaged_detections = []
    same_count_frame_number = 1
    previous_count = 0
    for detection in detections:
        if detection["detections"] != previous_count:
            if same_count_frame_number >= minimal_same_count_frame_number:
                averaged_detections.append({"detections":previous_count,"same_count_frame_number":same_count_frame_number})
                #averaged_detections.append(previous_count)
            same_count_frame_number = 1
        previous_count = detection["detections"]
        same_count_frame_number += 1
    return averaged_detections
            
    

def time_date_to_timestamp(time_dict, date_dict):
    timezone_string = "+00:00"
    timestamp = datetime.datetime.strptime(date_dict["year"] + '-' + date_dict["month"] + '-' + date_dict["day"]
                                  + 'T' + time_dict["hour"] + ':' + time_dict["minute"] + timezone_string, '%Y-%b-%dT%H:%M%z')
    print("Timestamp: " + str(timestamp))
    return timestamp


def date_to_timestamp(date_dict):
    timezone_string = "+00:00"
    timestamp = datetime.datetime.strptime(date_dict["year"] + '-' + date_dict["month"] + '-' + date_dict["day"],'%Y-%b-%d')
    print("Timestamp: " + str(timestamp))
    return timestamp