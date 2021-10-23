from subprocess import call
import sys
import os
data = sys.argv[1]

def schedule_detection(start_date, start_time):
    print("Start date: " + start_date)
    print("Start time: " + start_time)
    os.system("echo 'My python scheduler' >> /home/pi/Desktop/My_Server/PortableHumanRecognitionSystemWebApplication/backend/cron/croncontent" + 
    " | at " + start_time + " " + start_date)
    

schedule_detection(sys.argv[1], sys.argv[2])