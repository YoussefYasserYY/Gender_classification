from PIL import Image
# import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator as data_augment
import streamlit as st

st.title('Gender classification model')

train_datagen = data_augment(rescale = 1./255)

# train_datagen = ImageDataGenerator(rescale=1 / 255.0)

#data preprocessing and augmentation

traind = train_datagen.flow_from_directory(r"C:\Users\ghost\Desktop\gender_classification\test",
                                          target_size = (96, 96),
)

# loaded_model = pickle.load(open('C:/Users/ghost/Desktop/gender_classification/trained_model_gender.sav', 'rb'))




# (102,85,3)


# img = mpimg.imread('woman.jpg')

img = cv2.imread('download.jpg')

print(np.shape(img))

# img = img.thumbnail((96, 96))
# cv2.resize(img, None)

print(np.shape(img))


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a PIL Image:
    cam = Image.open(img_file_buffer)
    cam = np.resize(cam,(1,96,96,3))   
    print(np.shape(cam))
    x = loaded_model.predict(cam)
    print(x)
    if(x[0][0]>x[0][1]):
        # print('Female')
        st.write('Female')
    else:
        st.write('male')
        # print('Male')






