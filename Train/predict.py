import keras
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os, os.path
import cv2


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

def load_imgs(image_folder):
    imgs = {}
    valid_images = [".jpeg"]
    for f in os.listdir(image_folder):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        img_tensor = load_image(os.path.join(image_folder,f))
        imgs[f] = img_tensor

    return imgs


def predictone(img_path):
    new_image = load_image(img_path)
    model = load_model((os.environ['HOME']) + '/BDBI/Prototype/Train/weights/recyclesort_weights.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    pred = model.predict(new_image)
    classes = [line.rstrip('\n') for line in open((os.environ['HOME']) + "/chakra/BDBI/Prototype/Train/weights/recyclesort_classes.txt", "r")]

    for i in range(len(classes)):
        print ("class:", classes[i],"confidence:", pred[0][i])


def predictall(test_folder):
    test_images = load_imgs(test_folder)
    model = load_model((os.environ['HOME']) + "/chakra/BDBI/Prototype/Train/weights/recyclesort_weights.h5")
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam',
                  metrics=['accuracy'])
    classes = [line.rstrip('\n') for line in open((os.environ['HOME']) + "/chakra/BDBI/Prototype/Train/weights/recyclesort_classes.txt", "r")]
    for filename, image in test_images.items():
        pred = model.predict(image)
        pred = pred[0]
        max_index = np.argmax(pred)
        print("Actual:", filename, "Predicted", classes[max_index])

predictall((os.environ['HOME']) + '/chakra/BDBI/Prototype/Train/Test_images')
img_path = './chakra/BDBI/Prototype/Train/Test_images/testbottle1.jpeg'
