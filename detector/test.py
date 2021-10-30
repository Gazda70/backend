import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras
#import tensorflow_hub as hub

SSD_MOBILENET_V2_SAVED_MODEL_PATH="/home/pi/Desktop/My_Server/backend/models/"

#model_SSD = tf.keras.models.load_model(SSD_MOBILENET_V2_SAVED_MODEL_PATH)
#model_SSD = keras.models.load_model(SSD_MOBILENET_V2_SAVED_MODEL_PATH)

image = cv2.imread("/home/pi/Desktop/happy_people.jpeg")
frame = image.copy()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_resized = cv2.resize(frame_rgb, (300, 300))
img = frame_resized
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
model = tf.saved_model.load(SSD_MOBILENET_V2_SAVED_MODEL_PATH)


#print(list(loaded.signatures.keys()))  # ["serving_default"]

infer = model.signatures["serving_default"]
print(infer.structured_outputs) #{'output_0': TensorSpec(shape=(1, 2), dtype=tf.float32, name='output_0')}

# convert img to tf
x = tf.keras.preprocessing.image.img_to_array(img, dtype='uint8')
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis,...]).astype(dtype='uint8')
# (1,3,network_size, network_size)

labeling = infer(tf.constant(x))
print(labeling)
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