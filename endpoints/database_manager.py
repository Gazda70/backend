import pymongo

myclient = pymongo.MongoClient("mongodb://localhost:27017/")

mydb = myclient["database"]

mycol = mydb["Detections"]

'''
detectionPeriod = {"startDate": "03/11/2021", "startTime":"19:16", "endTime":"20:30", "neuralNetwork":"SSD Mobilenet v2 320x320", "certainityThreshold":"0.4"}

x = mycol.insert_one(detectionPeriod)

print(x.inserted_id)
'''

res = mycol.find().count()

print(res)

#for x in res: 
#    print(x)