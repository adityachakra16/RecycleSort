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
import matplotlib.pyplot as plt

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


def NN_model(base_model_name, freeze_layers, visaualize_layers):

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

    for layer in model.layers[:freeze_layers]:
        layer.trainable=False

    model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def train(model, epochs, train_data, validation_data, save_model, save_dir, visualize):

    step_size_train=train_data.n//train_data.batch_size
    if validation_data:
        step_size_validation = validation_data.n//validation_data.batch_size
    else:
        step_size_validation = None

    history = model.fit_generator(generator=train_data, validation_data = validation_data,
                        validation_steps = step_size_validation,
                       steps_per_epoch=step_size_train,
                       epochs=epochs)

    if save_model:
        model.save(save_dir)
        f = open(save_dir + "recyclesort_labels.txt", "w")
        for label in train_data.classes:
            f.write(label)
        f.close()

    if visualize:
        visualize_training_performance(history)

def visualize_training_performance(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

if __name__ == '__main__':

    train_data, validation_data = prepare_data('C:\\BDBI\\Prototype\\Train\\Train_images', 'C:\\BDBI\\Prototype\\Train\\Val_images', True)
    model = NN_model('MobileNetV2',  100, visaualize_layers = True)
    train(model, 1, train_data, validation_data, True, 'C:\\BDBI\\Prototype\\Train\\weights\\RecycleSort_weights.h5', visualize=True)
