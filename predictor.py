from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import os

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def predict_images(folder):
    '''A function that predicts the classes for images, taking a folder of images path as an input'''
    # image folder
    folder_path = folder
    # dimensions of images
    img_width, img_height = 64, 64
    # load all images into a list
    images = []
    for img in os.listdir(folder_path):
        #Uncomment this line of code if you will run it on OSX
        # if img == '.DS_Store':
        #     continue
        curr_img = image.load_img(folder_path+'/'+img, target_size=(img_width, img_height))
        x = image.img_to_array(curr_img)
        x = np.expand_dims(x, axis=0)
        images.append(x)
    # stack up images list to pass for prediction
    images = np.vstack(images)
    classes = loaded_model.predict_proba(images, batch_size=10)
    return classes



output = predict_images('test_images')
print(output)
