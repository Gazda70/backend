from threading import Thread
from picamera.array import PiRGBArray
from picamera import PiCamera
import time


class VideoStream:
    """Camera object that controls video streaming from the Picamera"""

    def __init__(self, resolution={"width":320, "height":320}, framerate=30):
        self.camera = PiCamera()
        self.camera.resolution = (resolution["width"], resolution["height"])
        self.camera.framerate = framerate
        self.rawCapture = PiRGBArray(self.camera, size = (resolution["width"], resolution["height"]))
        self.frame=None
        self.stopped=False
        time.sleep(0.1)

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            # If the camera is stopped, stop the thread
            if self.stopped:
                return

            # Otherwise, grab the next frame from the stream
            self.frame = frame.array
            
            self.rawCapture.truncate(0)

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True
        
'''
v = VideoStream()

v.start()

frame = v.read()

cv2.imshow("test", frame)

cv2.waitKey(0)

v.stop()
'''
