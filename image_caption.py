from flask import Flask

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
import pickle

app = Flask(__name__)



def caption_generator(image):
    with open('static/models/caption_tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)

    max_length = 21

    def idx_to_word(integer, tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def preprocess_image(image):
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        resized_image = cv2.resize(image, (224,224))

        # input_image_rgb=cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        # input_image_normalized = resized_image.astype(np.float32) / 255.0
        input_image_array = np.expand_dims(resized_image, axis=0)
        processed_image=preprocess_input(input_image_array)

        return processed_image


    # Load VGG16 model for feature extraction
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)

    def extract_vgg16_features(image_path):
        image = preprocess_image(image_path)
        extracted_features = model.predict(image.reshape((1, 224, 224, 3)))
        return extracted_features

    def predict_caption_tflite(interpreter, image_features, tokenizer, max_length):
        in_text = 'startsentence'
        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)

            input_data = {
                input_details[0]['index']: np.array(sequence).astype(np.float32),
                input_details[1]['index']: np.array(image_features).reshape((1, 4096)).astype(np.float32)
            }

            for idx, data in input_data.items():
                interpreter.set_tensor(idx, data)

            interpreter.invoke()
            yhat = interpreter.get_tensor(output_details[0]['index'])
            yhat = np.argmax(yhat)

            word = idx_to_word(yhat, tokenizer)
            if word is None or word == 'endsentence':
                break

            in_text += ' ' + word

        return in_text

    # Load the TensorFlow Lite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path='static/models/image_caption_model_cow.tflite')
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Extract VGG16 features
    image_features = extract_vgg16_features(image)

    # Predict caption using TFLite model
    caption = predict_caption_tflite(interpreter, image_features, tokenizer, max_length)
    
    return caption
