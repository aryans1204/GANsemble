import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def load_data():
    labels = ['glioma']
    image_size = 150
    train_folder_path = os.environ["TRAINING"]
    test_folder_path = os.environ["TESTING"]

    # Load the training dataset
    train_images = []
    train_labels = []
    for i, label in enumerate(labels):
        folder_path = os.path.join(train_folder_path, label)
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.resize(img, (image_size, image_size))
            train_images.append(img)
            train_labels.append(1)

    # Load the test dataset for use in model.evaluate
    test_images = []
    test_labels = []
    for i, label in enumerate(labels):
        folder_path = os.path.join(test_folder_path, label)
        for filename in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, filename))
            img = cv2.resize(img, (image_size, image_size))
            test_images.append(img)
            test_labels.append(1)

    # Convert the image data and label arrays to NumPy arrays
    X_train = np.array(train_images)
    Y_train = np.array(train_labels)
    X_test = np.array(test_images)
    Y_test = np.array(test_labels)
    # Convert the label arrays to integers
    Y_train = Y_train.astype(int)
    Y_test = Y_test.astype(int)

    #shuffle training dataset
    X_train,Y_train = shuffle(X_train,Y_train,random_state=101)
    # Define folder paths
    train_dir = os.path.join(train_folder_path, '')
    test_dir = os.path.join(test_folder_path, '')

    # Define image dimensions
    img_height = 128
    img_width = 128

    # Define ImageDataGenerator for training data
    train_datagen = ImageDataGenerator(
        # Data augmentation and rescaling parameters
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=1./255
    )

    # Define ImageDataGenerator for test data
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Generate data batches from folders; target size affects the size of imgs in data batches...
    #...but does not affect the actual image data that is loaded from the disk...
    #...actual img from disk (that is fed in model) is affected by image size = 150 in preprocessing
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_height, img_width),
        class_mode='binary')

    X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,test_size=0.1,random_state=101)
    Y_train = np.ones(Y_train.shape)
    Y_test = np.ones(Y_test.shape)
    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    load_data()
