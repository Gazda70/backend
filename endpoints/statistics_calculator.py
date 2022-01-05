class StatisticsCalculator:
    def min_detected_people(self, detections):
       minimal = self.max_detected_people(detections)
       for det in detections:
           if det["detections"] < minimal:
               minimal = det["detections"]
       return minimal
    
    
    def max_detected_people(self, detections):
       maximal = 0
       for det in detections:
           if det["detections"] > maximal:
               maximal = det["detections"]
       return maximal
    
    
    def arithmetic_average_detected_people(self, detections):
        people_count = 0
        for det in detections:
            people_count += det["detections"]
        if len(detections) != 0:
            return math.ceil(people_count/len(detections))
        return 0
        
        
    def divide_into_subcollections(self, detections, border_frame_number):
        averaged_detections = []
        count_frame_number = 1
        previous_count = 0
        summed_people_number = 0
        for detection in detections:
                summed_people_number = summed_people_number + detection["detections"]
                if count_frame_number >= border_frame_number:
                    averaged_detections.append(int(summed_people_number/count_frame_number))
                    count_frame_number = 1
                else
                    count_frame_number += 1
        if count_frame_number != 0:
            averaged_detections.append(int(summed_people_number/count_frame_number))
        return averaged_detections
