from detection_manager import DetectionManager
from database_manager import DatabaseManager
import sys, getopt
from datetime import datetime

try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["neuralNetworkType=", "detectionSeconds=", "obj_threshold=",
                       "video_resolution_width=", "video_resolution_height=", "framerate=", "detection_period_id="])
except getopt.GetoptError:
    sys.exit(2)
print("Hello")

print(opts)
print(args)
options = dict(opts)
print(options)
detection_manager = DetectionManager()
print("When starting detection, detection_period_id: " + str(options["--detection_period_id"]))
detection_manager.setupDetection(detection_period_id=str(options["--detection_period_id"]), neuralNetworkType=options["--neuralNetworkType"], detectionSeconds=int(options["--detectionSeconds"]),  obj_threshold=float(options["--obj_threshold"]),
                       video_resolution={"width":int(options["--video_resolution_width"]), "height":int(options["--video_resolution_height"])}, framerate=int(options["--framerate"]))
#detection_manager.detect()
