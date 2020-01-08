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

def prepare_data(train_folder, validation_folder, validation):
    datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

    train_generator=datagen.flow_from_directory(directory = train_folder,
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=32,
                                                     class_mode='categorical',
                                                     shuffle=True)
    if validation == True:
        validation_generator = datagen.flow_from_directory(directory=validation_folder,
                                                             target_size=(224,224),
                                                             batch_size=32,
                                                             shuffle = True,
                                                             classes = ['MetalCans', 'PlasticBottles'])

        return train_generator, validation_generator

    else:
        return train_generator, None


def NN_model(base_model_name, visaualize_layers):

    if base_model_name == 'MobileNetV2':
        base_model=MobileNetV2(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=Dense(1024,activation='relu')(x) #dense layer 2
        x=Dense(512,activation='relu')(x) #dense layer 3
        preds=Dense(2,activation='softmax')(x) #final layer with softmax activation


        model=Model(inputs=base_model.input,outputs=preds)
    else:
        print("Only supports MobileNetV2 currently")
        return

    if visaualize_layers == True:
        for i,layer in enumerate(model.layers):
          print(i,layer.name)

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def train(model, epochs, train_data, validation_data, freeze_layers):

    for layer in model.layers[:freeze_layers]:
        layer.trainable=False

    step_size_train=train_data.n//train_data.batch_size
    if validation_data:
        step_size_validation = validation_data.n//validation_data.batch_size
    else:
        step_size_validation = None

    history = model.fit_generator(generator=train_data, validation_data = validation_data,
                        validation_steps = step_size_validation,
                       steps_per_epoch=step_size_train,
                       epochs=epochs)

if __name__ == '__main__':

    train_data, validation_data = prepare_data('C:\\BDBI\\Prototype\\Train\\Train_images', 'C:\\BDBI\\Prototype\\Train\\Val_images', True)
    model = NN_model('MobileNetV2', visaualize_layers = True)
    train(model, 1, train_data, validation_data, 100)
