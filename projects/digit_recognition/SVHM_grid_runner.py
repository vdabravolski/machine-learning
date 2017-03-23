"""
Implemetation of random search for SVHM CNN.
"""
import pickle
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
import numpy as numpy
import os
import datetime
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from ioi_metric import iou_metric_func


def data():
    test_data = "./data/SVHM/test_processed/dataset.p"
    test_data = pickle.load(open(test_data, 'rb'))

    img_rows = numpy.shape(test_data[0][1])[0]
    img_cols = numpy.shape(test_data[0][1])[1]
    img_depth = numpy.shape(test_data[0][1])[2]

    X_test = numpy.zeros(shape=(len(test_data), img_rows, img_cols, img_depth))

    train_data = "./data/SVHM/train_processed/dataset.p"
    train_data = pickle.load(open(train_data, 'rb'))
    X_train = numpy.zeros(shape=(len(train_data), img_rows, img_cols, img_depth))

    length_train = numpy.zeros(shape=(len(train_data), 6))
    length_test = numpy.zeros(shape=(len(test_data), 6))

    coord_train = numpy.zeros(shape=(len(train_data), 20))
    coord_test = numpy.zeros(shape=(len(test_data), 20))

    first_train = numpy.zeros(shape=(len(train_data), 11))
    second_train = numpy.zeros(shape=(len(train_data), 11))
    third_train = numpy.zeros(shape=(len(train_data), 11))
    forth_train = numpy.zeros(shape=(len(train_data), 11))
    fifth_train = numpy.zeros(shape=(len(train_data), 11))

    first_test = numpy.zeros(shape=(len(test_data), 11))
    second_test = numpy.zeros(shape=(len(test_data), 11))
    third_test = numpy.zeros(shape=(len(test_data), 11))
    forth_test = numpy.zeros(shape=(len(test_data), 11))
    fifth_test = numpy.zeros(shape=(len(test_data), 11))

    for idx, el in enumerate(test_data):
        X_test[idx, :, :, :] = el[1]
        length_test[idx, :] = el[3]
        coord_test[idx, :] = numpy.asarray(el[2]).flatten()
        first_test[idx, :] = el[4][0]
        second_test[idx, :] = el[4][1]
        third_test[idx, :] = el[4][2]
        forth_test[idx, :] = el[4][3]
        fifth_test[idx, :] = el[4][4]

    for idx, el in enumerate(train_data):
        X_train[idx, :, :, :] = el[1]
        length_train[idx, :] = el[3]
        coord_train[idx, :] = numpy.asarray(el[2]).flatten()
        first_train[idx, :] = el[4][0]
        second_train[idx, :] = el[4][1]
        third_train[idx, :] = el[4][2]
        forth_train[idx, :] = el[4][3]
        fifth_train[idx, :] = el[4][4]

    Y_test = [length_test, first_test, second_test, third_test, forth_test, fifth_test, coord_test]
    Y_train = [length_train, first_train, second_train, third_train, forth_train, fifth_train, coord_train]
    return X_train, Y_train, X_test, Y_test

def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''

    numpy.random.seed(1337)
    nb_classes = 11
    sequence_length = 5
    img_rows, img_cols, img_depth = 64, 64, 3
    input_shape = (img_rows, img_cols, img_depth)

    batch_size = 128
    nb_epoch = 12
    nb_filters = 32
    pool_size = (2, 2)
    kernel_size = (3, 3)
    cls_weights = [1., 1., 1., 1., 1., 1., 1.]


    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                           border_mode='valid', input_shape=input_shape)(main_input)
    shared = Activation('relu')(shared)
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(shared)
    shared = Activation('relu')(shared)
    shared = MaxPooling2D(pool_size=pool_size)(shared)
    # shared = Dropout(0.25)(shared)
    shared = Dropout({{uniform(0, 1)}})(shared)
    shared = Flatten()(shared)
    shared = Dense(128)(shared)
    shared = Activation('relu')(shared)
    shared = Dropout(0.5)(shared)


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

    #7. Digit boxes coordinates regresssion
    coord_regr = Dense(20, name="coord_regr")(shared)

    # model compilation and training
    model = Model(input=[main_input], output=[length_cls, first_cls, second_cls, third_cls, forth_cls, fifth_cls, coord_regr])

    model.compile(loss={'length_cls': 'categorical_crossentropy', 'first_cls': 'categorical_crossentropy',
                        'second_cls': 'categorical_crossentropy', 'third_cls': 'categorical_crossentropy',
                        'forth_cls': 'categorical_crossentropy', 'fifth_cls': 'categorical_crossentropy',
                        'coord_regr': 'mean_squared_error'},
                  optimizer='adadelta',
                  metrics={'length_cls': 'accuracy', 'first_cls': 'accuracy',
                           'second_cls': 'accuracy', 'third_cls': 'accuracy',
                           'forth_cls': 'accuracy', 'fifth_cls': 'accuracy',
                           'coord_regr': iou_metric_func}, loss_weights=cls_weights)

    directory = "output/" + str(datetime.datetime.now())
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpointer = ModelCheckpoint(filepath=directory + "/weights.hdf5", verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=directory, histogram_freq=0, write_graph=True, write_images=False)
    model.fit([X_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=([X_test], Y_test), callbacks=[checkpointer, tensorboard])

    score, acc = model.evaluate(X_test, Y_test, verbose=0)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))





