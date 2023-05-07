import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
import glob
import keras

from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAvgPool2D
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img, img_to_array
from keras.callbacks import ModelCheckpoint, EarlyStopping
from django.shortcuts import render

ROOT_DIR = "media/skin_can"
number_of_images = {}

def index(request):
    if request.method == "POST":
        file = request.FILES["file"].read()
        with open("media/uploads/image.jpg", 'br+') as f:
            f.write(file)
        model = load_model("media/model/bestmodel.h5")
        path = "media/uploads/image.jpg"
        img = load_img(path, target_size=(224, 224))
        input_arr = img_to_array(img)/255
        input_arr.shape
        input_arr = np.expand_dims(input_arr, axis=0)
        pred = (model.predict(input_arr)[0][0] >= 0.003).astype("int32")

        if (pred == 0):
            result = "image shows no signs of cancer"
        elif (pred == 1):
            result = "the image shows signs of cancer"
        else:
            result = "image is unpredictable"
        return render(request, 'index.html', {'result': result})
    return render(request, 'index.html')
