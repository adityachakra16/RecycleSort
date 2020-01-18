import keras
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor

img_path = 'C:\\BDBI\\Prototype\\Train\\Test_images\\testbottle1.jpeg'

new_image = load_image(img_path)
model = load_model('C:\\BDBI\\Prototype\\Train\\weights\\recyclesort_weights.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
pred = model.predict(new_image)
classes = [line.rstrip('\n') for line in open("C:\\BDBI\\Prototype\\Train\\weights\\recyclesort_classes.txt", "r")]

for i in range(len(classes)):
    print ("class:", classes[i],"confidence:", pred[0][i])
