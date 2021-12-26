import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from neural_networks_data import NEURAL_NETWORKS

class Detector:
    def __init__(self, model_name):
        self.model = cv2.dnn.readNet(NEURAL_NETWORKS[model_name]["model"],
                        config=NEURAL_NETWORKS[model_name]["config"],
                        framework=NEURAL_NETWORKS[model_name]["framework"])
        self.class_names = NEURAL_NETWORKS[model_name]["class_names"]
        self.img_width = NEURAL_NETWORKS[model_name]["img_width"]
        self.img_height = NEURAL_NETWORKS[model_name]["img_height"]
        
    
    def detect(self, image, img_width, img_height):
        #model_SSD = tf.keras.models.load_model(SSD_MOBILENET_V2_SAVED_MODEL_PATH)
        #model_SSD = keras.models.load_model(SSD_MOBILENET_V2_SAVED_MODEL_PATH)
        #frame = image.copy()
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #img = cv2.resize(frame_rgb, (img_width, img_height))
        '''
        print("frame_resized: ")
        print(frame_resized.shape)
        frame_suited=np.expand_dims(frame_resized, axis=0)
        print("frame_suited: ")
        print(frame_suited.shape)
        #summary=model_SSD.summary()
        #print(summary)
        output = model_SSD.predict(frame_suited)
        print(outputff)
        # load model
        '''
        #print(list(loaded.signatures.keys()))  # ["serving_default"]
        '''
        infer = self.model.signatures["serving_default"]#{'output_0': TensorSpec(shape=(1, 2), dtype=tf.float32, name='output_0')}

        # convert img to tf
        x = tf.keras.preprocessing.image.img_to_array(image, dtype='uint8')
        x = tf.keras.applications.mobilenet.preprocess_input(
            x[tf.newaxis,...]).astype(dtype='uint8')
        # (1,3,network_size, network_size)

        labeling = infer(tf.constant(x))
        return labeling
        '''
        '''
        print(labeling)

        img_width=1000
        img_height=1000
        final_image = cv2.resize(img, (1000, 1000))

        for box, score, detection_class in zip(labeling['detection_boxes'][0], labeling['detection_scores'][0], labeling['detection_classes'][0]):
            if score > 0.1:
                font = cv2.FONT_HERSHEY_SIMPLEX
                print('detection_class: ' + str(detection_class.numpy()))
                #print('detection_class type: ' + type(detection_class))
                cv2.putText(final_image, category_map[int(detection_class)], (int(box[1]*img_width), int(box[0]*img_height)), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
                final_image = cv2.rectangle(final_image, (int(box[1]*img_width), int(box[0]*img_height)), (int(box[3]*img_width), int(box[2]*img_height)), (0, 255, 0), 2)

        cv2.imshow("Window", final_image)
        
        for sr, box in zip(score_rescaled, boxes):
            print('PREDICTED LABEL: ' + LABELS[box.label])
            xmin = self.adjust_minmax(int(box.xmin * IMAGE_W), IMAGE_W)
            ymin = self.adjust_minmax(int(box.ymin * IMAGE_H), IMAGE_H)
            xmax = self.adjust_minmax(int(box.xmax * IMAGE_W), IMAGE_W)
            ymax = self.adjust_minmax(int(box.ymax * IMAGE_H), IMAGE_H)
            if LABELS[box.label] == 'person':
                predicted_boxes.append(
                    {'name': LABELS[box.label], 'score': box.get_score(),
                                                    'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
                                                    '''
        # Use keypoints if available in detections

        #predict_class = np.argmax(labeling['output_0'].numpy())
        #print(predict_class ) # int, depends on your task -- mine was img classfication
        #self.model_SSD.setInput(cv2.dnn.blobFromImage(frame_resized, size=(img_w, img_h), swapRB=True))
        #output = self.model_SSD.forward()
        '''
        x = tf.keras.preprocessing.image.img_to_array(image, dtype='uint8')
        x = tf.keras.applications.mobilenet.preprocess_input(
        x[tf.newaxis,...]).astype(dtype='uint8')
        output = self.model.predict(cv2.dnn.blobFromImage(frame_resized, size=(img_w, img_h), swapRB=True))
        print(output)
        return output
        '''
        '''
        detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        detector_output = detector(frame_suited)
        class_ids = detector_output["detection_classes"]
        print(class_ids)
        '''
        '''
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path="/home/pi/Desktop/My_Server/backend/models/tf_lite/model.tflite")
        '''
        '''
        self.model.allocate_tensors()

        # Get input and output tensors.
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        print("input_details: ")
        print(input_details)
        print("input_blob: ")
        #input_blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True)
        x = tf.keras.preprocessing.image.img_to_array(image, dtype='uint8')
        x = tf.keras.applications.mobilenet.preprocess_input(
        x[tf.newaxis,...]).astype(dtype='uint8')
        #print(input_blob.shape)
        #input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
        #input_data = np.array(cv2.dnn.blobFromImage(frame_resized, size=(300, 300), swapRB=True), dtype=np.uint8)
        self.model.set_tensor(input_details[0]['index'], x)

        self.model.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = self.model.get_tensor(output_details[0]['index'])
        print(output_data)
        '''
        
        xml_file = '/home/pi/Desktop/My_Server/backend/endpoints/haarcascades/haarcascade_upperbody.xml'
        classifier = cv2.CascadeClassifier(xml_file)
        
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces_rect = classifier.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=9)
        return faces_rect


    def detect_cv2(self, img):
        COLORS = np.random.uniform(0, 255, size=(len(self.class_names), 3))
        image = img
        image_height, image_width, _ = image.shape
        # create blob from image
        blob = cv2.dnn.blobFromImage(image=image, size=(self.img_width, self.img_height), mean=(104, 117, 123), swapRB=True)
        # start time to calculate FPS
        start = time.time()
        self.model.setInput(blob)
        output = self.model.forward()       
        # end time after detection
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)
        number_of_people = 0
        # loop over each of the detections
        for detection in output[0, 0, :, :]:
            # extract the confidence of the detection
            confidence = detection[2]
            # draw bounding boxes only if the detection confidence is above...
            # ... a certain threshold, else skip
            if confidence > .4:
                # get the class id
                class_id = detection[1]
                if class_id == 1:
                    number_of_people += 1
                # map the class id to the class
                if int(class_id) < len(self.class_names):
                    class_name = self.class_names[int(class_id)-1]
                    color = COLORS[int(class_id)]
                    # get the bounding box coordinates
                    box_x = detection[3] * image_width
                    box_y = detection[4] * image_height
                    # get the bounding box width and height
                    box_width = detection[5] * image_width
                    box_height = detection[6] * image_height
                    # draw a rectangle around each detected object
                    cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)
                    # put the class name text on the detected object
                    cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    # put the FPS text on top of the frame
                    cv2.putText(image, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Actual frame", img)
        cv2.waitKey(1) & 0xFF
        return number_of_people
