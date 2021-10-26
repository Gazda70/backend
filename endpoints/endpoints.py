# coding=utf-8

from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
#from detection_manager import DetectionManager
import json
import time
import ast
from detection_scheduler import schedule_detection
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

@app.route('/predictions')
def get_predictions():
    pass
    '''
    detection_manager = DetectionManager()
    response_body = detection_manager.get_detection_data()
    response_body_json = json.dumps(response_body)
    response = make_response(response_body_json, 200)
    response.mimetype = "application/json"
    print('Response body: ')
    print(response_body_json)
    return response
    '''
@app.route('/setup', methods=['GET','POST'])
def setup_detection():
    req = request.get_json()
    print("startDate: ")
    startDate = ast.literal_eval(req["startDate"])
    startTime = ast.literal_eval(req["startTime"])
    print(startTime["hour"] + ":" + startTime["minute"] + " " + startTime["halfOfDay"])
    print(startDate["month"] + " " + startDate["day"])
    schedule_detection(startDate["month"] + " " + startDate["day"],
                       startTime["hour"] + ":" + startTime["minute"])
    response_body = {
      "startDay": "Wed",
      "endDay": "Thurs",
      "startTime": "12:34",
      "endTime": "15:20",
      "totalDetections": []
    }
    response_body_json = json.dumps(response_body)
    response = make_response(response_body_json, 200)
    response.mimetype = "application/json"
    return response