import keras
from keras import backend as K
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam


base_model=MobileNetV2(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation


model=Model(inputs=base_model.input,outputs=preds)


for i,layer in enumerate(model.layers):
  print(i,layer.name)


for layer in model.layers[:100]:
    layer.trainable=False

datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=datagen.flow_from_directory('Train_images/',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

validation_generator = datagen.flow_from_directory(directory='Val_images/',
                                                     target_size=(224,224),
                                                     batch_size=32,
                                                     shuffle = True,
                                                     classes = ['MetalCans', 'PlasticBottles'])

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
step_size_validation = validation_generator.n//validation_generator.batch_size
model.fit_generator(generator=train_generator, validation_data = validation_generator,
                    validation_steps = step_size_validation,
                   steps_per_epoch=step_size_train,
                   epochs=5)
