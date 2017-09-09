import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import pickle
from shutil import copy2   
import cv2

import sklearn
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Conv2D, BatchNormalization, Cropping2D
from keras.utils import np_utils


def read_training_data(logfile, image_directory, steering_threshold, steering_correction, straight_steering_drop_prob): 
    """
    Load training data from 'logfile'.

    Prepend directory to the image file names in the input file.
    Drop frames with steering value less than steering_threshold 
    (too close to straight line driving) with probably given by 
    straigh_steering_drop_prob.
    For the left / right images, apply steering_correction.

    File is CSV file in format of CarND simulator data.

    """
    image_paths = []
    steering = []

    df = pd.read_csv(logfile)
    for _, row in df.iterrows():
        # If steering angle is too close to zero, discard the frame with 90% probability
        if abs(row['steering']) < steering_threshold and np.random.uniform() < straight_steering_drop_prob:
            continue

        # Add the 3 image paths - correct left right with steering_correction
        image_paths.append(os.path.join(image_directory, row['center'].strip()))
        steering.append(row['steering'])

        image_paths.append(os.path.join(image_directory, row['left'].strip()))
        steering.append(row['steering'] + steering_correction)

        image_paths.append(os.path.join(image_directory, row['right'].strip()))
        steering.append(row['steering'] - steering_correction)

    return (image_paths, steering)


def make_model(args):
    """
    Make neural net model from args
    """
    model = Sequential()

    model.add(Conv2D(24, (5, 5), strides=(2,2), activation=args.cnn_activation,
        input_shape=args.input_shape))
    model.add(Conv2D(36, (5, 5), strides=(2,2), activation=args.cnn_activation))
    model.add(Conv2D(48, (5, 5), strides=(2,2), activation=args.cnn_activation))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation=args.cnn_activation))
    model.add(Conv2D(64, (3, 3), activation=args.cnn_activation))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(100, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(50, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(10, kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dropout(0.5))

    model.add(Dense(1, kernel_regularizer=keras.regularizers.l2(0.001)))

    # Use Adam optimizer
    # Start with default params
    opt=keras.optimizers.Adam()

    # Use mean squared error loss for regression
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
    model.summary()

    return model


def process_image(image, crop):
    """
    Process the raw image.
    1. Normalize
    2. Crop
    We do not resize - resizing degraded driving performance
    """
    # Read the cropping params
    (top, bot, left, right) = crop

    normalized_image = image / 255. - 0.5
    cropped_image = normalized_image[top:bot,left:right,:]
    return cropped_image


def generator(samples, args):
    """
    Generate batch_size of samples per iteration.
    """
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, args.batch_size):
            batch_samples = samples[offset:offset+args.batch_size]

            images = []
            angles = []
            for image_path, steering in batch_samples:
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processed_image = process_image(rgb_image, args.crop)

                images.append(processed_image)
                angles.append(steering)
                # Flip image horizontally
                images.append(cv2.flip(processed_image, 1))
                angles.append(-steering)

            inputs = np.array(images)
            outputs = np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)


def train_model(image_paths, steering, args):
    """
    Train & save model based on arguments
    """
    samples = list(zip(image_paths, steering))
    train_samples, validation_samples = train_test_split(samples, test_size=args.validation_split)

    print('Training Samples: ' + str(len(train_samples)))
    print('Validation Samples: ' + str(len(validation_samples)))

    train_generator = generator(train_samples, args)
    validation_generator = generator(validation_samples, args)

    # Create model
    model = make_model(args)

    # Train
    history=model.fit_generator(train_generator,
                        # steps are doubled because we add flipped samples
                        steps_per_epoch = len(train_samples)*2 // args.batch_size,
                        validation_data=validation_generator,
                        validation_steps=len(validation_samples)*2 // args.batch_size,
                        epochs=args.epochs
                        )

    return (model, history)


def do_run(args):
    """
    Do a training run with supplied arguments.
    """
    image_paths, steering = read_training_data(args.logfile, 
                                            args.image_directory, 
                                            args.steering_threshold, 
                                            args.steering_correction,
                                            args.straight_steering_drop_prob)

    (model, history) = train_model(image_paths, steering, args)
    save_model_files(model, history, args)

    return (image_paths, steering, model, history)


def save_model_files(model, history, args):
    """
    Save the model and other files
    """
    if not os.path.exists(args.save_directory):
        os.mkdir(args.save_directory)
    # Save model and history
    model.save(os.path.join(args.save_directory, 'model.h5'))
    with open(os.path.join(args.save_directory, 'history.pkl'), 'wb') as f:
        pickle.dump(history.history, f, protocol=pickle.HIGHEST_PROTOCOL)
    # Save args
    with open(os.path.join(args.save_directory, 'params.pkl'), 'wb') as f:
        pickle.dump(vars(args), f, protocol=pickle.HIGHEST_PROTOCOL)
    # Copy this file for model changes
    copy2('model.py', args.save_directory)


def plot_training_history(history_history):
    """
    Plot the Keras training history.

    Input is Keras history.history dictionary (which can be pickled)
    """
    # Accuracy
    plt.figure(figsize=(10,5))
    plt.plot(history_history['acc'])
    plt.plot(history_history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.show()

    # Loss
    plt.figure(figsize=(10,5))
    plt.plot(history_history['loss'])
    plt.plot(history_history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()


def load_training_images(image_paths, args):
    """
    Load all image files into memory in cases where this is possible.
    """
    images = []
    for file in image_paths:
        image = cv2.imread(file)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        processed_image = process_image(rgb_image, args.crop)
        images.append(processed_image)
    return images


def test_model(model, image_paths, steering):
    """
    Compute model accuracy using image files and steering angles
    """
    images = load_training_images(image_paths)
    X = np.array(images)
    y = np.array(steering)
    y_pred = model.predict(X)
    return np.average(np.abs(y_pred - y))

