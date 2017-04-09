from __future__ import print_function

"""
Implemetation of random search for SVHM CNN.
"""
import pickle
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Convolution2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as numpy
import os
import datetime
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from ioi_metric import iou_metric_func
from svhm_image_generator import custom_image_generator

def data():
    """
    returns the keras image generators
    """
    batch_size = 128


    #Image transformation
    # TODO: P2 currently featurewise center and whitening are not calculated. First we need to do fit()
    featurewise_center = False
    samplewise_center = True
    featurewise_std_normalization = False
    samplewise_std_normalization = True
    color_mode = 'rgb'
    shuffle = True
    rescale = 1./255


    test_gen = ImageDataGenerator(featurewise_std_normalization=featurewise_std_normalization,
                                  samplewise_std_normalization=samplewise_std_normalization,
                                  featurewise_center=featurewise_center, samplewise_center=samplewise_center,
                                  rescale=rescale)
    test_dir_gen = test_gen.flow_from_directory('data/SVHM/test_generator', target_size=(64, 64), batch_size=batch_size,
                                      class_mode="sparse", shuffle=shuffle , color_mode=color_mode)
    test_gen = custom_image_generator(test_dir_gen, labels_pickle="data/SVHM/test_generator/dataset.p")

    train_gen = ImageDataGenerator(featurewise_std_normalization=featurewise_std_normalization,
                                   samplewise_std_normalization=samplewise_std_normalization,
                                   featurewise_center=featurewise_center, samplewise_center=samplewise_center,
                                   rescale=rescale)

    train_dir_gen = train_gen.flow_from_directory('data/SVHM/train_generator', target_size=(64, 64), batch_size=batch_size,
                                      class_mode="sparse", shuffle=shuffle , color_mode=color_mode)
    train_gen = custom_image_generator(train_dir_gen, labels_pickle="data/SVHM/train_generator/dataset.p")

    return train_gen, test_gen




def model(train_gen, test_gen):
    '''
    Model providing function:

    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''

    #numpy.random.seed(8994)
    nb_classes = 11
    sequence_length = 5
    img_rows, img_cols, img_depth = 64, 64, 3
    input_shape = (img_rows, img_cols, img_depth)

    nb_epoch = 12
    nb_filters = {{choice([32, 64, 128])}}
    pool_size = (2, 2)
    kernel_size = (3, 3)
    cls_weights = [1., 1., 1., 1., 1., 1., 2*{{uniform(0, 1)}}]
    #optimizer = {{choice(['rmsprop', 'adam', 'sgd', 'adadelta'])}} # TODO: leads to inf loss
    optimizer = 'adadelta'

    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                           border_mode='valid', input_shape=input_shape)(main_input)
    shared = Activation('relu')(shared)
    shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1])(shared)
    shared = Activation('relu')(shared)
    shared = MaxPooling2D(pool_size=pool_size)(shared)

    # Conditional extra convolutional layer
    if conditional({{choice(['two', 'three'])}}) == 'three':
        shared = Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                               border_mode='valid', input_shape=input_shape)(main_input)
        shared = Activation('relu')(shared)
        shared = MaxPooling2D(pool_size=pool_size)(shared)

    dropout_coef = {{uniform(0, 1)}}
    shared = Dropout(dropout_coef)(shared)
    shared = Flatten()(shared)
    shared = Dense(128)(shared)
    shared = Activation('relu')(shared)
    shared = Dropout(dropout_coef)(shared)

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
                  optimizer=optimizer,
                  metrics={'length_cls': 'accuracy', 'first_cls': 'accuracy',
                           'second_cls': 'accuracy', 'third_cls': 'accuracy',
                           'forth_cls': 'accuracy', 'fifth_cls': 'accuracy',
                           'coord_regr': iou_metric_func}, loss_weights=cls_weights)

    directory = "output/" + str(datetime.datetime.now())
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpointer = ModelCheckpoint(filepath=directory + "/weights.hdf5", verbose=1, save_best_only=True)
    tensorboard = TensorBoard(log_dir=directory, histogram_freq=0, write_graph=True, write_images=False)
    # Todo: P2 add EarlyStopping callback
    # Todo: P3 add LearningRateSchedule callback
    # TODO: P2 batch size are not randomized
    model.fit_generator(train_gen, samples_per_epoch=33401, nb_epoch=nb_epoch, verbose=1, validation_data=test_gen,
                        nb_val_samples=13068, callbacks=[checkpointer, tensorboard])


    score = model.evaluate_generator(test_gen, val_samples=13068)

    print('Test accuracy:', score)
    return {'loss': score[0], 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())

    train_gen, test_gen = data()
    print("Evalutation of best performing model:")
    #print(best_model.evaluate(X_test, Y_test))
    print(best_model.evaluate_generator(test_gen, val_samples=13068))
    print("Model parameters:")
    print(best_run)





