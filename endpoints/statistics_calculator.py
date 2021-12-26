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
        
        
    def divide_into_same_people_count_collections(self, detections, minimal_same_count_frame_number):
        averaged_detections = []
        same_count_frame_number = 1
        previous_count = 0
        for detection in detections:
            if detection["detections"] != previous_count:
                if same_count_frame_number >= minimal_same_count_frame_number:
                    averaged_detections.append({"detections":previous_count,"same_count_frame_number":same_count_frame_number})
                same_count_frame_number = 1
            previous_count = detection["detections"]
            same_count_frame_number += 1
        return averaged_detections
