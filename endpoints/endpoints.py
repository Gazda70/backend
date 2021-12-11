# coding=utf
import cv2
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
#from detection_manager import DetectionManager
import json
import time
import math
import ast
from detection_scheduler import schedule_detection
from database_manager import DatabaseManager, date_to_timestamp, divideIntoSamePeopleCountCollections
#from Detection import detect
# ... other import statements ...

# creating the Flask application
app = Flask(__name__)
CORS(app)
'''
def get_detector_occupancy(filename="/home/pi/Desktop/Backend/PortableHumanRecognitionSystemWebApplication/backend/DetectionState/IfDetectorOccupied.txt"):
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0)
        return val

def set_detector_occupancy(detector_occupancy=0, filename="/home/pi/Desktop/Backend/PortableHumanRecognitionSystemWebApplication/backend/DetectionState/IfDetectorOccupied.txt"):
    with open(filename, "a+") as f:
        f.seek(0)
        f.truncate()
        f.write(str(detector_occupancy))   

@app.route('/check_detection_state')
def check_if_ongoing_detection():
    is_ongoing = "true"
    detector_occupancy = get_detector_occupancy()
    print("det state: " + str(detector_occupancy))
    if detector_occupancy == 0:
        is_ongoing = "false"
    return is_ongoing
'''

@app.route('/predictions', methods=['GET','POST'])
def get_predictions():
    req = request.get_json()
    print("Request:")
    print(req)
    print("Date: ")
    print(ast.literal_eval(req["startDate"]))
    #print(date_to_timestamp(ast.literal_eval(req["startDate"])))
    database_manager = DatabaseManager()
    if req["mode"] == "period":
        results = database_manager.findDetectionPeriodsForGivenDateRange(date_to_timestamp(ast.literal_eval(req["startDate"])),
                                                                         date_to_timestamp(ast.literal_eval(req["endDate"])))
    elif req["mode"] == "single_day":
        results = database_manager.findDetectionPeriodsForGivenDate(date_to_timestamp(ast.literal_eval(req["startDate"])))
    detection_period_list = list(results)
    print("Detection period list: " + str(detection_period_list))
    detection_period_stats = []
    for det_per in detection_period_list:
        print("Object id: " + str(det_per.get('_id')))
        detections = database_manager.findAllDetectionsForGivenDetectionPeriod(str(det_per.get('_id')))
        detections_list = list(detections)
        print(len(detections_list))
        print(len(detections_list))
        averaged_detections = divideIntoSamePeopleCountCollections(detections_list, 2)
        print("averaged_detections")
        print(averaged_detections)
        people_min = database_manager.minDetectedPeople(averaged_detections)
        people_max = database_manager.maxDetectedPeople(averaged_detections) 
        start_time = det_per["start_time"].strftime("%m%d%Y %H:%M:%S").split(' ')[1]
        print("Start time: " + start_time)
        detection_period_stats.append({"start_time":start_time, "people_min":people_min, 'people_max':people_max})
    
    response_body = {"detection_period_stats":detection_period_stats}
    print("Response body: ")
    print(response_body)
    response_body_json = json.dumps(response_body)
    response = make_response(response_body_json, 200)
    response.mimetype = "application/json"
    print('Response body: ')
    print(response_body_json)
    return response


@app.route('/setup', methods=['GET','POST'])
def setup_detection():
    #print("Python version: " + platform.python_version())
    cv2.imread("/home/pi/Desktop/happy_people.jpeg")
    req = request.get_json()
    print("req: ")
    print(req)
    print("start_date: ")
    start_date = ast.literal_eval(req["startDate"])
    start_time = ast.literal_eval(req["startTime"])
    end_time = ast.literal_eval(req["endTime"])
    print(start_date)
    print("start_time: ")
    print(start_time)
    print("end_time")
    print(end_time)
    schedule_detection(start_date, start_time, end_time)
                       
    response_body = {
        "response":"Request OK"
    }
    response_body_json = json.dumps(response_body)
    response = make_response(response_body_json, 200)
    response.mimetype = "application/json"
    return response