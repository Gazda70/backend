from subprocess import call
import sys
import os
data = sys.argv[1]

def schedule_detection(start_date, start_time):
    print("Start date: " + start_date)
    print("Start time: " + start_time)
    os.system("echo 'Detecting now' >> /home/pi/Desktop/My_Server/backend/test" + 
    " | at " + start_time)
    

schedule_detection(sys.argv[1], sys.argv[2])