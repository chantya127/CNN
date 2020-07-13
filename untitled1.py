# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:57:42 2020

@author: User
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import cv2
import os


import numpy as np


from keras.preprocessing import image
import keras
from tensorflow.python.keras.backend import set_session

import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

# Load your trained model
#from tensorflow.keras.models import load_model

session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    keras.backend.set_session(session)
    model = tf.keras.models.load_model(MODEL_PATH)
    model._make_predict_function()      
   

# Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    test_image = image.load_img(img_path, target_size=(256, 256))
    test_image = image.img_to_array(test_image)
    test_image = cv2.resize(test_image,(256,256))
    #image.append(img)
    test_image = test_image / 255
    test_image = test_image.reshape(-1,256,256,3)
    
    #preds = ''
    
    print('hiii1')

    # Preprocessing the image
    #test_image = np.expand_dims(test_image,axis = 0)
    
    #session = keras.backend.get_session()
    #init = tf.global_variables_initializer()
    #session.run(init)
    
    #with graph.as_default():
     #   set_session(sess)
      #  result = model.predict(test_image)
    with session.graph.as_default(): 
        
        keras.backend.set_session(session)
        predict = model.predict(test_image)
        predict = np.argmax(predict)
    #graph = tf.get_default_graph()
    #result = model.predict(test_image)
    
    print("hii2")
    
    if predict == 0:
        cmd = 'Covid_19'
    elif predict ==1:
        cmd = 'Normal'
    elif predict ==2:
        cmd = 'Viral Pneumonia'    
    else:
        cmd = 'Bacterial Pneumonia'
    print(cmd)   
        
    return cmd


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        
        print("hii0")

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   
        #result = str(pred_class[0][0][1])               
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)