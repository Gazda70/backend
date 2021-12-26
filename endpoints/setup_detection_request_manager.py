import time
import math
import ast
from detection_scheduler import schedule_detection

class SetupDetectionRequestManager:
    def manage_request(self, req):
        start_date = ast.literal_eval(req["startDate"])
        start_time = ast.literal_eval(req["startTime"])
        end_time = ast.literal_eval(req["endTime"])
        schedule_detection(start_date, start_time, end_time)
