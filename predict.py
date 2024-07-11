import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import json
from PIL import Image
import argparse
import tensorflow_hub as hub

parser = argparse.ArgumentParser(description='takes the path of the image of a flower, and the model and classifies it')
parser.add_argument('image_path', type=str)
parser.add_argument('model_name', type=str)
parser.add_argument('--top_k',type=int, default=5)
parser.add_argument('--category_names', type=str, default='label_map.json')
arg = parser.parse_args()

print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
    reloaded_keras_model = tf.keras.models.load_model(arg.model_name)
reloaded_keras_model.summary()

with open(arg.category_names, 'r') as f:
    class_names = json.load(f)

image_size = 224

def predict(img_path, model, top_k):
    im = Image.open(img_path)
    image_tensor = (tf.image.resize(tf.convert_to_tensor(im, dtype=tf.float32), (image_size, image_size))
    / 255)
    image_tensor = image_tensor[None, :, :, :]
    ans = model.predict(image_tensor).squeeze()
    probs = np.sort(ans)
    classes = np.argsort(ans)
    return probs[::-1][0:top_k], classes[::-1][0:top_k]

probs, classes = predict(arg.image_path, reloaded_keras_model, arg.top_k)

print("for the image you gave me i predict the following:")
for i in range(0, arg.top_k):
    flower_name = class_names[f'{classes[i]}'] 
    print(f'there is a chance that it is {flower_name} with a expectancy = {probs[i]:.4f}%')