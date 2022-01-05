#!/bin/bash

python3.7 /home/pi/Desktop/My_Server/backend/endpoints/scheduled_detection.py --detection_period_id=61d5ba3d74fece0fd6ccde78 --neural_network_type=SSD_Mobilenet_v2_320x320 --detection_seconds=120 --obj_threshold=0.3 --box_overlap_threshold=0.5 --video_resolution_width=320 --video_resolution_height=320 --framerate=30