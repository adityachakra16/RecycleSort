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
from tensorflow import lite
import numpy as np
from IPython.display import Image
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from glob import glob
import os
import argparse
TF_CPP_MIN_LOG_LEVEL=2

#TODO: Perform automatic training, validation split from one folder
def prepare_data(train_folder, validation_folder):
    '''
    params:
    train_folder: Folder containing training dataset
    validation_folder: Folder containing validation dataset

    purpose: Create seperate generator objects for training and validation data.

    return:
    Dataset Generator
    '''
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator=datagen.flow_from_directory(directory = train_folder,
                                                     target_size=(224,224),
                                                     color_mode='rgb',
                                                     batch_size=32,
                                                     class_mode='categorical',
                                                     shuffle=True)

    if validation_folder:
        validation_generator = datagen.flow_from_directory(directory=validation_folder,
                                                             target_size=(224,224),
                                                             batch_size=32,
                                                             shuffle = True)


        return train_generator, validation_generator

    else:
        return train_generator, None


def NN_model(base_model_name, freeze_layers, visaualize_layers):
    '''
    params:
    base_model_name: Name of the model used for feature extraction
    freeze_layers: Number of layers to freeze from the top.
    visaualize_layers: print all the layers of the model

    return: model with pre-trained weights from imagenet and custom added layers
    '''
    if base_model_name == 'MobileNetV2':
        base_model=MobileNetV2(weights='imagenet',include_top=False)

        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x=Dense(1024,activation='relu')(x)
        x=Dense(1024,activation='relu')(x)
        x=Dense(512,activation='relu')(x)
        preds=Dense(2,activation='softmax')(x)


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

def train(model, epochs, train_data, validation_data, visualize):
    '''
    params:
    model: model to be trained
    epochs: Number of epochs to be trained for
    train_data: training data generator object
    validation_data: validation data generator object
    visualize: visualize training performance

    return: trained model
    '''

    step_size_train=train_data.n//train_data.batch_size
    if validation_data:
        step_size_validation = validation_data.n//validation_data.batch_size
    else:
        step_size_validation = None

    history = model.fit_generator(generator=train_data, validation_data = validation_data,
                        validation_steps = step_size_validation,
                       steps_per_epoch=step_size_train,
                       epochs=epochs)

    if visualize:
        visualize_training_performance(history)

    return model


def saving_model(save_dir, model, classes, tflite_model):
    #TODO: Find a way to directly save model in tflite
    '''
    params:
    save_dir: Directory to save model in
    model: trained model
    classes: class labels
    tflite_model: save model in tflite format
    '''

    model.save(save_dir)
    f = open(save_dir + "recyclesort_labels.txt", "w")
    for label in classes:
        f.write(label + "\n")
    f.close()

    if tflite_model:
        converter = lite.TFLiteConverter.from_keras_model_file(save_dir + "recyclesort_labels.txt")
        tfmodel = converter.convert()
        open ("recyclesort_weights.tflite" , "wb").write(tfmodel)

def visualize_training_performance(history):
    '''
    params:
    history: training history
    '''

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



parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--train_dir', type=str, default='C:\\BDBI\\all_dataset\\TrashNet_dataset\\Current_categories\\Train\\Train',
                    help='Directory where training data is stored')
parser.add_argument('--val_dir', type=str, default='',
                    help='Directory where validation data is stored')
parser.add_argument('--model', type=str, default='MobileNetV2',
                    help='Name of Base Model')
parser.add_argument('--freeze', type=int, default=155,
                    help='Number of layers from top to freeze while training')
parser.add_argument('--visualize_model', type=bool, default=False,
                    help='Visualize training performance')
parser.add_argument('--visualize_performance', type=bool, default=False,
                    help='Visualize training performance')
parser.add_argument('--epochs', type=int, default=3,
                    help='Number of epochs')
parser.add_argument('--save_model', type=bool, default=True,
                    help='Save model?')
parser.add_argument('--save_dir', type=str, default='C:\\BDBI\\Prototype\\Train\\weights\\recyclesort_weights.h5',
                    help='Directory to save model')
parser.add_argument('--tflite', type=bool, default=False,
                    help='save in .tflite format')

if __name__ == '__main__':
    args = parser.parse_args()
    train_data, validation_data = prepare_data(args.train_dir, args.val_dir)
    NNmodel = NN_model(args.model,  args.freeze, args.visualize_model)
    trained_model = train(NNmodel, args.epochs, train_data, validation_data, args.visualize_performance)
    if args.save_model:
        saving_model(args.save_dir, trained_model, os.listdir(args.train_dir), args.tflite)
