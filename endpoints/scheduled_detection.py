from detection_manager import DetectionManager
from database_manager import DatabaseManager
import sys, getopt
from datetime import datetime
import time
print("HERE")
try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["neural_network_type=", "detection_seconds=", "obj_threshold=", "box_overlap_threshold=",
                       "video_resolution_width=", "video_resolution_height=", "framerate=", "detection_period_id="])
except getopt.GetoptError:
    file = open("test_file", "w")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    file.write("Failed: " + str(current_time))
    file.close()
    start_time = time.time()
    sys.exit(2)


options = dict(opts)

detection_manager = DetectionManager()
detection_manager.setupDetection(detection_period_id=str(options["--detection_period_id"]), neural_network_type=options["--neural_network_type"], detection_seconds=int(options["--detection_seconds"]),  obj_threshold=float(options["--obj_threshold"]),
                       box_overlap_threshold=float(options["--box_overlap_threshold"]), video_resolution={"width":int(options["--video_resolution_width"]), "height":int(options["--video_resolution_height"])}, framerate=int(options["--framerate"]))