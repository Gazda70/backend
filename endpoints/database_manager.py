import pymongo

class DatabaseManager:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")

        self.db = self.client["PeopleCounterDatabase"]

        self.detectionPeriodCollection = self.db["DetectionPeriod"]
        
        self.detectionCollection = self.db["Detection"]
        
    def insertDetectionPeriod(self, startDate, startTime, endTime, neuralNetworkType, detectionSeconds,  obj_threshold,
                       video_resolution, framerate):
        return_value = self.detectionPeriodCollection.insert_one({"startDate":startDate, "startTime":startTime, "endTime":endTime, "detectionSeconds":detectionSeconds,
                                                                  "neuralNetworkType":neuralNetworkType, "obj_threshold":obj_threshold,
                       "video_resolution":video_resolution, "framerate":framerate})
        return return_value.inserted_id
        
    def insertDetection(self, time, detections, detectionPeriodId):
        self.detectionCollection.insert_one({"time":time, "detections":detections, "detectionPeriodId":detectionPeriodId})
        
    def findDetectionsForDetectionPeriod(self, detectionPeriodId):
        self.detectionCollection.find({"detectionPeriodId":detectionPeriodId})
        
        

'''
detectionPeriod = {"startDate": "03/11/2021", "startTime":"19:16", "endTime":"20:30", "neuralNetwork":"SSD Mobilenet v2 320x320", "certainityThreshold":"0.4"}

x = mycol.insert_one(detectionPeriod)

print(x.inserted_id)
'''

#for x in res: 
#    print(x)