# coding=utf
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
import json
from get_statistics_request_manager import GetStatisticsRequestManager
#from setup_detection_request_manager import SetupDetectionRequestManager

app = Flask(__name__)
CORS(app)

@app.route('/predictions', methods=['GET','POST'])
def get_predictions():
    req = request.get_json()
    request_manager = GetStatisticsRequestManager()
    response_body = request_manager.manage_request(req)
    response_body_json = json.dumps(response_body)
    response = make_response(response_body_json, 200)
    response.mimetype = "application/json"
    return response


@app.route('/setup', methods=['GET','POST'])
def setup_detection():
    req = request.get_json()
    request_manager = SetupDetectionRequestManager()
    request_manager.manage_request(req)
    response_body = {
        "response":"Request OK"
    }
    response_body_json = json.dumps(response_body)
    response = make_response(response_body_json, 200)
    response.mimetype = "application/json"
    return response
    