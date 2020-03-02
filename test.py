import cv2
import glob
import numpy as np
import pandas as pd
from keras import backend as K
import keras
from keras.models import Sequential,Model
from keras.applications.densenet import DenseNet121, preprocess_input
import tensorflow as tf
from keras.layers import Dense,Input,Conv2D,MaxPooling2D,Activation,Flatten,Dropout,Cropping2D,Lambda,GlobalAveragePooling2D,BatchNormalization,AveragePooling2D
import pickle

EPOCHS = 50
BS = 8
SIZE = 300 ## Resize factor
TEST_SIZE = 0.2
label_size=4


base_model = DenseNet121(include_top=False, weights=None,
                         input_shape=(SIZE, SIZE, 3), classes=15)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)

predictions = Dense(4, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('model_weights.h5')

with open("output_dict.pkl","rb") as file:
    output_dict=pickle.load(file)

test_images=glob.glob("test/*.jpg")
wrong_pred=0
for img_path in test_images:
    img=cv2.imread(img_path)
    img=cv2.resize(img,(SIZE,SIZE))
    pred=model.predict(np.array([img]))
    label_indice=np.argmax(pred)
    print("Image_name: {}, predicted output: {}".format(img_path.split("/")[-1],output_dict[label_indice]))
