import datetime
import time
import re

detectionTimeString = "12:21"
findings = re.search("^(?:([01]?\d|2[0-3]):([0-5]?\d))?$", detectionTimeString)

timeValues = []
if findings:
    print("Matched")
    timeValues = detectionTimeString.split(':')

for timeVal in timeValues:
    print("Time value: " + timeVal)


x = datetime.datetime.now().time()
hours = int(timeValues[0]) - x.hour
minutes = int(timeValues[1]) - x.minute
print("Hours: " + str(hours))
print("Minutes: " + str(minutes))
if minutes < 0:
    minutes = 60 - minutes
    hours -= 1

if hours < 0:
    print("End time must be grater that start time !")
start_time = time.time()
print("Start time: " + str(start_time))
seconds = 10
while True:
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > seconds:
        print("Finished iterating in: " + str(int(elapsed_time))  + " seconds")
        break
'''
now = datetime.now()
print("Now: " + str(now))
detection_time = datetime.strptime(detectionTimeString, '%I:%M')
print("Detection time: " + str(detection_time))
detection_end = now + detection_time
print("Detection end: " + str(detection_end))
'''