import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("/home/pi/Desktop/My_Server/backend/models") # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('/home/pi/Desktop/My_Server/backend/models/tf_lite/ssd_mobilenet_v2.tflite', 'wb') as f:
  f.write(tflite_model)