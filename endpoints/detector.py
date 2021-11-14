import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras

SSD_MOBILENET_V2_SAVED_MODEL_PATH="/home/pi/Desktop/My_Server/backend/models/"
category_map = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}
img_width=300
img_height=300
image = cv2.imread("/home/pi/Desktop/happyPeople.jpg")

class Detector:
    def __init__(self, model_path, category_map):
        self.model = tf.saved_model.load(model_path)
        self.category_map = category_map
        
    
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

        infer = self.model.signatures["serving_default"]#{'output_0': TensorSpec(shape=(1, 2), dtype=tf.float32, name='output_0')}

        # convert img to tf
        x = tf.keras.preprocessing.image.img_to_array(image, dtype='uint8')
        x = tf.keras.applications.mobilenet.preprocess_input(
            x[tf.newaxis,...]).astype(dtype='uint8')
        # (1,3,network_size, network_size)

        labeling = infer(tf.constant(x))
        return labeling
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

        #output = model_SSD.predict(cv2.dnn.blobFromImage(frame_resized, size=(img_w, img_h), swapRB=True))
        #print(output)
        '''
        detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        detector_output = detector(frame_suited)
        class_ids = detector_output["detection_classes"]
        print(class_ids)
        '''
        '''
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path="/home/pi/Desktop/My_Server/backend/models/tf_lite/model.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        print("input_details: ")
        print(input_details)
        print("input_blob: ")
        input_blob = cv2.dnn.blobFromImage(frame_resized, size=(300, 300), swapRB=True)
        print(input_blob.shape)
        #input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
        #input_data = np.array(cv2.dnn.blobFromImage(frame_resized, size=(300, 300), swapRB=True), dtype=np.uint8)
        interpreter.set_tensor(input_details[0]['index'], frame_suited)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        '''
