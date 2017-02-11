'''Trains CNN on the synthetical digit sequences, created from MNIST dataset.

Features:
- length of the sequence to train CNN to identify digits from noise.
- digits represented as 1-hot.
'''

from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K
from mnist_data_preps import generate_sequences
from matplotlib import pyplot as plt

batch_size = 128
nb_classes = 11  # include 10 digits and "-1" which designates empty space in digit sequence.
nb_epoch = 12
sequence_length = 5

# input image dimensions
img_rows, img_cols = 28, (28 * sequence_length)
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
weights_path = "tmp/vanilla_cnn_weights.hdf5"

# the data, shuffled and split between train and test sets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

X_train_orig, Y_train, Y_train_length = generate_sequences(train_images, train_labels, out_size=np.shape(train_images)[0],
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

train_targets = [length_cls_train, first_cls_train, second_cls_train, third_cls_train, forth_cls_train, fifth_cls_train]
test_targets = [length_cls_test, first_cls_test, second_cls_test, third_cls_test, forth_cls_test, fifth_cls_test]
cls_weights = [1., 1., 1., 1., 1., 1.]


def create_vanilla_CNN(cls_weights):
    # Shared feature layers which will be used for multiple classifiers.
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                           border_mode='valid', input_shape=input_shape)(main_input)
    shared = Activation('relu')(shared)
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(shared)
    shared = Activation('relu')(shared)
    shared = MaxPooling2D(pool_size=pool_size)(shared)
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

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'], loss_weights=cls_weights)

    return model


def train_model(model, train_inputs, train_targets, test_inputs, test_targets, use_trained=False):
    if use_trained:
        model = model
        model.load_weights(weights_path)
    else:
        checkpointer = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True)  # TODO: add relative path
        model.fit([train_inputs], train_targets, batch_size=batch_size, nb_epoch=nb_epoch,
                  verbose=1, validation_data=([test_inputs], test_targets), callbacks=[checkpointer])

    return model


def evaluate_model(model, eval_inputs, eval_targets, verbose=1):
    score = model.evaluate(eval_inputs, eval_targets, verbose=verbose)
    print('Length accuracy:', score[7])
    print('First digit accuracy:', score[8])
    print('Second digit accuracy:', score[9])
    print('Third digit accuracy:', score[10])
    print('Forth digit accuracy:', score[11])
    print('Fifth digit accuracy:', score[12])

def predict_sequence(model, pred_input):
    predictions = model.predict(pred_input)

    pred_legth = np.argmax(predictions[0][0])
    pred_sequence = ""

    for i in xrange(1, (pred_legth+1)):
        pred_sequence += str(np.argmax(predictions[i][0]))

    return pred_sequence



new_model = create_vanilla_CNN(cls_weights)
trained_model = train_model(model=new_model, train_inputs=X_train,
                            train_targets=train_targets, test_inputs=X_test,test_targets=test_targets, use_trained=True)

evaluate_model(trained_model,X_test,test_targets)

### Sampling predictions
fig = plt.figure()

for i in range(1,10):
    random_index = np.random.choice(10000)
    prediction = predict_sequence(trained_model, pred_input=X_test[np.newaxis, random_index])

    # Create subplot
    a = fig.add_subplot(3, 3, i)
    imgplot = plt.imshow(X_test_orig[random_index])
    a.set_title('Predicted:'+prediction)
plt.show()