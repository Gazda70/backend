from detection_manager import DetectionManager
import sys, getopt

try:
    opts, args = getopt.getopt(sys.argv[1:], "", ["neuralNetworkType=", "detectionSeconds=", "obj_threshold=",
                       "video_resolution_width=", "video_resolution_height=", "framerate="])
except getopt.GetoptError:
    sys.exit(2)

print(opts)
print(args)
options = dict(opts)
print(options)
detection_manager = DetectionManager()
detection_manager.setupDetection(neuralNetworkType=options["--neuralNetworkType"], detectionSeconds=int(options["--detectionSeconds"]),  obj_threshold=float(options["--obj_threshold"]),
                       video_resolution={"width":int(options["--video_resolution_width"]), "height":int(options["--video_resolution_height"])}, framerate=int(options["--framerate"]))

detection_manager.detect()