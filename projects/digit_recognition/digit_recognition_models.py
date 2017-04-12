'''Trains CNN on the synthetical digit sequences, created from MNIST dataset.

Features:
- length of the sequence to train CNN to identify digits from noise.
- digits represented as 1-hot.
'''

from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K
from mnist_data_preps import generate_sequences
from matplotlib import pyplot as plt
import matplotlib as mpl
import pickle
import tensorflow as tf
import datetime
import os
from svhm_image_proc import draw_BBOX

tf.python.control_flow_ops = tf  # to overcome some compatibility issues between keras and tf

choice = 'train'  # decide what to do: train, evaluate or predict
image_type = 'grayscale'

np.random.seed(1666)  # for reproducibility
batch_size = 128
nb_classes = 11  # include 10 digits and "-1" which designates empty space in digit sequence.
nb_epoch = 24 # TODO: normal is 12.
sequence_length = 5

# input image dimensions
# img_rows, img_cols = 28, (28 * sequence_length) # this is for MNIST

if image_type == 'grayscale':
    img_rows, img_cols, img_depth = 64, 64, 1
    input_shape = (img_rows, img_cols, img_depth)
elif image_type == 'rgb':
    img_rows, img_cols, img_depth = 64, 64, 3  # this is for SVHM. None - grayscal, 3 - RGB
    input_shape = (img_rows, img_cols, img_depth)

# number of convolutional filters to use
nb_filters = 32  # used to be 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
pretrained_weights = "tmp/best_weights.hdf5"


def prepare_MNIST_data():
    # the data, shuffled and split between train and test sets
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    X_train_orig, Y_train, Y_train_length = generate_sequences(train_images, train_labels,
                                                               out_size=np.shape(train_images)[0],
                                                               max_length=sequence_length)
    X_test_orig, Y_test, Y_test_length = generate_sequences(test_images, test_labels, out_size=np.shape(test_images)[0],
                                                            max_length=sequence_length)

    if K.image_dim_ordering() == 'th':
        X_train = X_train_orig.reshape(X_train_orig.shape[0], 1, img_rows, img_cols)
        X_test = X_test_orig.reshape(X_test_orig.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train_orig.reshape(X_train_orig.shape[0], img_rows, img_cols, 1)
        X_test = X_test_orig.reshape(X_test_orig.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    # Classifiers' inputs and targets
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # 1. Length classifier
    length_cls_train = Y_train_length
    length_cls_test = Y_test_length

    # 2. First digit classifier
    first_cls_train = Y_train[:, 0, :]
    first_cls_test = Y_test[:, 0, :]

    # 3.Second digit classifier
    second_cls_train = Y_train[:, 1, :]
    second_cls_test = Y_test[:, 1, :]

    # 4. Third digit classifier
    third_cls_train = Y_train[:, 2, :]
    third_cls_test = Y_test[:, 2, :]

    # 5. Forth digit classifier
    forth_cls_train = Y_train[:, 3, :]
    forth_cls_test = Y_test[:, 3, :]

    # 6. First digit classifier
    fifth_cls_train = Y_train[:, 4, :]
    fifth_cls_test = Y_test[:, 4, :]

    train_targets = [length_cls_train, first_cls_train, second_cls_train, third_cls_train, forth_cls_train,
                     fifth_cls_train]
    test_targets = [length_cls_test, first_cls_test, second_cls_test, third_cls_test, forth_cls_test, fifth_cls_test]

    return X_train, train_targets, X_test, test_targets, input_shape


def prepare_SVHM_data(input_shape=input_shape):
    test_data = "./data/SVHM/test_grayscale/dataset.p"
    test_data = pickle.load(open(test_data, 'rb'))

    train_data = "./data/SVHM/train_grayscale/dataset.p"
    train_data = pickle.load(open(train_data, 'rb'))

    # Images
    if input_shape[2]==1:
        X_train = np.zeros(shape=(len(train_data), img_rows, img_cols))
        X_test = np.zeros(shape=(len(test_data), img_rows, img_cols))
    elif input_shape[2]==3:
        X_train = np.zeros(shape=(len(train_data), img_rows, img_cols, img_depth))
        X_test = np.zeros(shape=(len(test_data), img_rows, img_cols, img_depth))

    # 1. Length
    length_train = np.zeros(shape=(len(train_data), 6))
    length_test = np.zeros(shape=(len(test_data), 6))

    # 2. Coordinates
    coord_train = np.zeros(
        shape=(len(train_data), 20))  # here 20 = 5 (max number of digits) * 4 (coordinates for each digit)
    coord_test = np.zeros(shape=(len(test_data), 20))

    # 3. Digits
    first_train = np.zeros(shape=(len(train_data), 11))
    second_train = np.zeros(shape=(len(train_data), 11))
    third_train = np.zeros(shape=(len(train_data), 11))
    forth_train = np.zeros(shape=(len(train_data), 11))
    fifth_train = np.zeros(shape=(len(train_data), 11))

    first_test = np.zeros(shape=(len(test_data), 11))
    second_test = np.zeros(shape=(len(test_data), 11))
    third_test = np.zeros(shape=(len(test_data), 11))
    forth_test = np.zeros(shape=(len(test_data), 11))
    fifth_test = np.zeros(shape=(len(test_data), 11))

    for idx, el in enumerate(test_data):
        #X_test[idx, :, :, :] = el[1]
        X_test[idx] = el[1]
        length_test[idx, :] = el[3]
        coord_test[idx, :] = np.asarray(el[2]).flatten()
        first_test[idx, :] = el[4][0]
        second_test[idx, :] = el[4][1]
        third_test[idx, :] = el[4][2]
        forth_test[idx, :] = el[4][3]
        fifth_test[idx, :] = el[4][4]

    for idx, el in enumerate(train_data):
        #X_train[idx, :, :, :] = el[1]
        X_train[idx] = el[1]
        length_train[idx, :] = el[3]
        coord_train[idx, :] = np.asarray(el[2]).flatten()
        first_train[idx, :] = el[4][0]
        second_train[idx, :] = el[4][1]
        third_train[idx, :] = el[4][2]
        forth_train[idx, :] = el[4][3]
        fifth_train[idx, :] = el[4][4]

    if input_shape[2]==1:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, img_depth)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, img_depth)


    test_targets = [length_test, first_test, second_test, third_test, forth_test, fifth_test, coord_test]
    train_targets = [length_train, first_train, second_train, third_train, forth_train, fifth_train, coord_train]

    return X_train, train_targets, X_test, test_targets, input_shape


def prepare_SVHM_test_data():
    test_data = "./data/SVHM/test_processed/dataset.p"
    test_data = pickle.load(open(test_data, 'rb'))

    X_test = np.zeros(shape=(len(test_data), img_rows, img_cols, img_depth))
    length_test = np.zeros(shape=(len(test_data), 6))
    coord_test = np.zeros(shape=(len(test_data), 20))

    first_test = np.zeros(shape=(len(test_data), 11))
    second_test = np.zeros(shape=(len(test_data), 11))
    third_test = np.zeros(shape=(len(test_data), 11))
    forth_test = np.zeros(shape=(len(test_data), 11))
    fifth_test = np.zeros(shape=(len(test_data), 11))

    for idx, el in enumerate(test_data):
        X_test[idx, :, :, :] = el[1]
        length_test[idx, :] = el[3]
        coord_test[idx, :] = np.asarray(el[2]).flatten()
        first_test[idx, :] = el[4][0]
        second_test[idx, :] = el[4][1]
        third_test[idx, :] = el[4][2]
        forth_test[idx, :] = el[4][3]
        fifth_test[idx, :] = el[4][4]

    test_targets = [length_test, first_test, second_test, third_test, forth_test, fifth_test, coord_test]
    input_shape = (img_rows, img_cols, img_depth)

    return X_test, test_targets, input_shape


def sample_SVHM_test_data(nb_samples=10):
    """
    Get some data for predictions.
    :return:
    """

    test_data = "./data/SVHM/test_processed/dataset.p"
    test_data = pickle.load(open(test_data, 'rb'))

    random_index = np.random.choice(len(test_data), nb_samples)

    X_sample = np.zeros(shape=(nb_samples, img_rows, img_cols, img_depth))
    length_sample = np.zeros(shape=(nb_samples, 6))
    coord_sample = np.zeros(shape=(nb_samples, 20))

    first_sample = np.zeros(shape=(nb_samples, 11))
    second_sample = np.zeros(shape=(nb_samples, 11))
    third_sample = np.zeros(shape=(nb_samples, 11))
    forth_sample = np.zeros(shape=(nb_samples, 11))
    fifth_sample = np.zeros(shape=(nb_samples, 11))

    for i, idx in enumerate(random_index):
        X_sample[i, :, :, :] = test_data[idx][1]
        length_sample[i, :] = test_data[idx][3]
        coord_sample[i, :] = np.asarray(test_data[idx][2]).flatten()
        first_sample[i, :] = test_data[idx][4][0]
        second_sample[i, :] = test_data[idx][4][1]
        third_sample[i, :] = test_data[idx][4][2]
        forth_sample[i, :] = test_data[idx][4][3]
        fifth_sample[i, :] = test_data[idx][4][4]

    test_targets = [length_sample, first_sample, second_sample, third_sample, forth_sample, fifth_sample, coord_sample]

    return X_sample, test_targets


def synth_MNIST_model(cls_weights):
    # Shared feature layers which will be used for multiple classifiers.
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                           border_mode='valid', input_shape=input_shape)(main_input)
    shared = Activation('relu')(shared)
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(shared)
    shared = Activation('relu')(shared)
    shared = MaxPooling2D(pool_size=pool_size)(shared)

    # # TODO: note that this is an extra layer, Not all weights will work with it.
    # shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
    #                        border_mode='valid', input_shape=input_shape)(main_input)
    # shared = Activation('relu')(shared)
    # shared = MaxPooling2D(pool_size=pool_size)(shared)

    shared = Dropout(0.25)(shared)
    shared = Flatten()(shared)
    shared = Dense(128)(shared)
    shared = Activation('relu')(shared)
    shared = Dropout(0.5)(shared)

    # 6 different classifiers.
    # 1. Length classifier
    length_cls = Dense((sequence_length + 1))(shared)  # to account for sequence length + 1
    length_cls = Activation('softmax', name="length")(length_cls)

    # 2. First digit classifier
    first_cls = Dense(nb_classes)(shared)
    first_cls = Activation('softmax', name="first_digit")(first_cls)

    # 3. Second digit classifier
    second_cls = Dense(nb_classes)(shared)
    second_cls = Activation('softmax', name="second_digit")(second_cls)

    # 4. Third digit classifier
    third_cls = Dense(nb_classes)(shared)
    third_cls = Activation('softmax', name="third_digit")(third_cls)

    # 5. Forth digit classifier
    forth_cls = Dense(nb_classes)(shared)
    forth_cls = Activation('softmax', name="forth_digit")(forth_cls)

    # 6. Fifth digit classifier
    fifth_cls = Dense(nb_classes)(shared)
    fifth_cls = Activation('softmax', name="fifth_digit")(fifth_cls)

    # model compilation and training
    model = Model(input=[main_input], output=[length_cls, first_cls, second_cls, third_cls, forth_cls, fifth_cls])

    model.compile(
        loss={'categorical_crossentropy': [length_cls, first_cls, second_cls, third_cls, forth_cls, fifth_cls]},
        optimizer='adadelta',
        metrics=['accuracy'], loss_weights=cls_weights)

    return model


def SVHM_model(cls_weight):
    # Shared feature layers which will be used for multiple classifiers.
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                           border_mode='valid', input_shape=input_shape)(main_input)
    shared = Activation('relu')(shared)
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(shared)
    shared = Activation('relu')(shared)
    shared = MaxPooling2D(pool_size=pool_size)(shared)
    shared = Dropout(0.25)(shared)

    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(shared)
    shared = Activation('relu')(shared)
    shared = Dropout(0.25)(shared)

    shared = MaxPooling2D(pool_size=pool_size)(shared)

    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(shared)
    shared = Activation('relu')(shared)
    shared = Dropout(0.25)(shared)


    shared = Flatten()(shared)
    shared = Dense(128)(shared)
    shared = Activation('relu')(shared)
    shared = Dropout(0.25)(shared)

    # 6 different classifiers.
    # 1. Length classifier
    length_cls = Dense((sequence_length + 1))(shared)  # to account for sequence length + 1
    length_cls = Activation('softmax', name="length_cls")(length_cls)

    # 2. First digit classifier
    first_cls = Dense(nb_classes)(shared)
    first_cls = Activation('softmax', name="first_cls")(first_cls)

    # 3. Second digit classifier
    second_cls = Dense(nb_classes)(shared)
    second_cls = Activation('softmax', name="second_cls")(second_cls)

    # 4. Third digit classifier
    third_cls = Dense(nb_classes)(shared)
    third_cls = Activation('softmax', name="third_cls")(third_cls)

    # 5. Forth digit classifier
    forth_cls = Dense(nb_classes)(shared)
    forth_cls = Activation('softmax', name="forth_cls")(forth_cls)

    # 6. Fifth digit classifier
    fifth_cls = Dense(nb_classes)(shared)
    fifth_cls = Activation('softmax', name="fifth_cls")(fifth_cls)

    # 7. Digit boxes coordinates regresssion
    coord_regr = Dense(20, name="coord_regr")(shared)

    # model compilation and training
    model = Model(input=[main_input],
                  output=[length_cls, first_cls, second_cls, third_cls, forth_cls, fifth_cls, coord_regr])

    model.compile(loss={'length_cls': 'categorical_crossentropy', 'first_cls': 'categorical_crossentropy',
                        'second_cls': 'categorical_crossentropy', 'third_cls': 'categorical_crossentropy',
                        'forth_cls': 'categorical_crossentropy', 'fifth_cls': 'categorical_crossentropy',
                        'coord_regr': 'mean_squared_error'},
                  optimizer='adadelta',
                  metrics={'length_cls': 'accuracy', 'first_cls': 'accuracy',
                           'second_cls': 'accuracy', 'third_cls': 'accuracy',
                           'forth_cls': 'accuracy', 'fifth_cls': 'accuracy',
                           'coord_regr': _iou_metric}, loss_weights=cls_weights)

    return model


def _iou_metric(y_true, y_pred, epsilon=1e-5, sequence_length=5):
    """ Inspired by: http://ronny.rest/tutorials/module/localization_001/intersect_of_union/

        Given two arrays `y_true` and `y_pred` where each row contains a bounding
        boxes for sequence of digits. By default, sequence length is 5. Each digit is represented by 4 numbers:
            [y1, x1, y2, x2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each sequence.

    Args:
        y_true:          (numpy array) sequence_length * each row containing [y1, x1, y2, x2] coordinates
        y_pred:          (numpy array) sequence_length * each row containing [y1, x1, y2, x2] coordinates
        epsilon:    (float) Small value to prevent division by zero
        sequence_length: (int) number of digits in the sequence

    Returns:
        (float) Sum of IoU for all digits in sequence
    """

    # Reshape the sequence coordinates which comes flatten from the regressor.
    y_true = K.reshape(y_true, [-1, 5, 4])
    y_pred = K.reshape(y_pred, [-1, 5, 4])

    K.print_tensor(y_true, "OLOLO")

    # COORDINATES OF THE INTERSECTION BOXES
    y1 = K.maximum(y_true[:, :, 0], y_pred[:, :, 0])
    x1 = K.maximum(y_true[:, :, 1], y_pred[:, :, 1])
    y2 = K.minimum(y_true[:, :, 2], y_pred[:, :, 2])
    x2 = K.minimum(y_true[:, :, 3], y_pred[:, :, 3])

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width = K.clip(width, 0, None)
    height = K.clip(height, 0, None)

    # area_overlap = width * height
    # area_overlap = K.prod(width * height)
    area_overlap = K.tf.multiply(width, height)

    # COMBINED AREAS
    area_a = (y_true[:, :, 2] - y_true[:, :, 0]) * (y_true[:, :, 3] - y_true[:, :, 1])
    area_b = (y_pred[:, :, 2] - y_pred[:, :, 0]) * (y_pred[:, :, 3] - y_pred[:, :, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    iou = K.mean(iou)  # reduce mean across all axis

    return iou


def train_model(model, train_inputs=None, train_targets=None, test_inputs=None, test_targets=None, use_trained=False):
    if use_trained:
        model = model
        model.load_weights(pretrained_weights)
    else:
        # create appropriate directory to store  weights and logs.
        directory = "output/" + str(datetime.datetime.now())
        if not os.path.exists(directory):
            os.makedirs(directory)

        checkpointer = ModelCheckpoint(filepath=directory + "/weights.hdf5", verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=directory, histogram_freq=0, write_graph=True, write_images=False)
        model.fit([train_inputs], train_targets, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=([test_inputs], test_targets), callbacks=[checkpointer, tensorboard])

    return model


def evaluate_model(model, eval_inputs, eval_targets, verbose=1):
    score = model.evaluate(eval_inputs, eval_targets, verbose=verbose)
    for idx in xrange(len(score)):
        print(model.metrics_names[idx] + ": " + str(score[idx]))


def predict_sequence(model, pred_input, return_coord=True):
    predictions = model.predict(pred_input)

    pred_legth = np.argmax(predictions[0][0])
    pred_sequence = ""

    for i in xrange(1, (pred_legth + 1)):
        pred_sequence += str(np.argmax(predictions[i][0]))

    if return_coord:
        pred_coord = (predictions[6])
        return pred_sequence, pred_coord

    return pred_sequence


# Training SVHM model

if choice == 'evaluate':
    X_test, test_targets, input_shape = prepare_SVHM_test_data()
    cls_weights = [1., 1., 1., 1., 1., 1., 1.]
    new_model = SVHM_model(cls_weights)
    new_model.load_weights(pretrained_weights)
    evaluate_model(new_model, X_test, test_targets)

elif choice == 'train':
    X_train, train_targets, X_test, test_targets, input_shape = prepare_SVHM_data()
    cls_weights = [1., 1., 1., 1., 1., 1., 1.]
    new_model = SVHM_model(cls_weights)
    trained_model = train_model(model=new_model, train_inputs=X_train,
                                train_targets=train_targets, test_inputs=X_test,
                                test_targets=test_targets, use_trained=False)
    evaluate_model(trained_model, X_test, test_targets)
elif choice == 'predict':
    nb_samples = 25
    X_sample, sample_targets = sample_SVHM_test_data(nb_samples=nb_samples)
    cls_weights = [1., 1., 1., 1., 1., 1., 1.]
    new_model = SVHM_model(cls_weights)
    trained_model = train_model(model=new_model, use_trained=True)

    ### Sampling predictions
    fig = plt.figure()

    mpl.rcParams['axes.titlesize'] = 'small'
    plt.axis('off')

    for idx, el in enumerate(X_sample):
        prediction, pred_coord = predict_sequence(trained_model, pred_input=el[np.newaxis])

        # Create subplot
        a = fig.add_subplot(5, 5, (idx + 1))

        imgplot = plt.imshow(draw_BBOX(el, pred_coord.reshape((5, 4))))

        true_sequence = ''
        for i in xrange(np.argmax(sample_targets[0][idx])):
            label = str(np.argmax(sample_targets[i + 1][idx]))
            true_sequence += label
        a.set_title('Predicted:' + prediction + "  label:" + true_sequence)

    plt.show()
