import numpy as np

np.random.seed(999)
import matplotlib.pyplot as pyplot
from keras.utils import to_categorical
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.layers import Dense
from keras.layers import LSTM, Activation, Dropout
import pickle
from keras.optimizers import SGD
from keras.layers import Dense, Merge
from keras.layers.normalization import BatchNormalization
import math
import pandas as pd
import random
import json
import os
import uuid

TIMESERIES_FOLDER = "data/timeseries/"


def get_ts_ticker_data(ticker, output_shape, train_test_ratio=0.7, sliding_window=True, classification=True):
    """method to do necessary data massage
    sliding window - if True, then input_dim[0] sliding windows is created;
    output_shape:
        defined the output shape of X samples;
        Y outputs is (output_shape[0], 1), where 1 - is exactly one value in the end of each time series.
    classification - defined if Y should contain boolean value (1 if stock grow, 0 if fell.
                    if classification = False, then stock price (float) is returned.
    """

    file = TIMESERIES_FOLDER + "{0}_df.p".format(ticker)
    with open(file, "rb") as file:
        ticker_df = pickle.load(file)

    data_df = pd.DataFrame(columns=ticker_df.columns)
    target_df = pd.DataFrame(columns=['diff_bool', 'value'])

    if sliding_window:  # implementation of sliding window. Each window is moving by 1 element.

        if len(output_shape) != 3:
            raise ValueError("If you choose sliding window parameter"
                             "please specify 3 output dimensions for data."
                             "Now you passed {0}".format(len(output_shape)))

        # Explicitly define output dimensions
        batch_size = output_shape[0]
        timesteps = output_shape[1]
        features = output_shape[2]

        for idx in range(ticker_df.shape[0] - timesteps-1):

            data_sample = ticker_df[idx: idx + timesteps]
            diff = int(ticker_df.close[idx + timesteps + 1] > ticker_df.close[idx + timesteps])
            target_sample = pd.DataFrame([[diff, ticker_df.close[idx + timesteps+1]]],
                                         columns=['diff_bool', 'value'])

            data_df = data_df.append(data_sample)
            target_df = target_df.append(target_sample, ignore_index=True)

    else:

        raise NotImplemented("Shaping time-series data without sliding windows are to be implemented if needed.")

        if len(output_shape) != 2:
            raise ValueError("Please specify output with shape (batch_size, feature_size)"
                             "Now you passed {0}".format(len(output_shape)))


        # Explicitly define output dimensions
        batch_size = output_shape[0]
        features = output_shape[1]


    if classification:
        Y = target_df.diff_bool
    else:
        Y = target_df.value


    data_df = data_df.reset_index(drop=True)
    X = data_df.loc[:,['open', 'high', 'low', 'close', 'volume']]  # TODO: reduced number of features are used.

    X_train_index = int(X.shape[0] * train_test_ratio)
    Y_train_index = int(Y.shape[0] * train_test_ratio)

    Y_train = Y.values[:Y_train_index]
    Y_test = Y.values[(Y_train_index + 1):]

    X_train = X.values[:X_train_index, :]
    X_test = X.values[(X_train_index + 1):, :]



    # TODO: when resizing, we are trimming a lot of data to fit into the batch*timestep size.
    X_train = _resize_data_for_batches(X_train, batch_size * timesteps)
    X_test = _resize_data_for_batches(X_test, batch_size * timesteps)
    X_train = np.reshape(X_train, (-1, timesteps, features))
    X_test = np.reshape(X_test, (-1, timesteps, features))

    Y_train = Y_train[:X_train.shape[0]]
    Y_test = Y_test[:X_test.shape[0]]


    # resize to batches


    print("Input data shape: \n X train: {0}, Y train: {1} \n X test: {2}, Y test: {3}".
          format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
    return X_train, Y_train, X_test, Y_test


def get_complex_model():
    """a model which combine several branches with various inputs"""

    # A dictionary with specific config. Will be used to track experiments settings.
    model_config = {'name': 'complex_model'}
    model_config['optimizer'] = 'adam'
    #model_config['loss'] = 'binary_crossentropy'
    model_config['loss'] = 'mean_squared_error'

    # TODO: in future add here nlp and fundamentals branches
    model, ts_config = ts_model_branch()
    model_config['ts_config'] = ts_config
    model.compile(loss=model_config['loss'], optimizer=model_config['optimizer'])

    return model, model_config

def ts_model_branch():
    c = {'name': 'ts_branch'}
    c['batch_size'] = batch_size
    c['timesteps'] = timesteps
    c['features'] = 5  # TODO: this is a hardcoded value. Need to come up with not hardcoded alternative.
    c['lstm_shapes'] = [100, 50, 20]  # TODO: #2 element is inline with with batch size.
    c['activation'] = 'linear'
    c['go_backwards'] = True
    c['dense_shapes'] = [20, 1]
    c['return_sequence'] = True
    c['stateful'] = True
    c['output'] = 1

    ## Explanations of return_sequence and TimeDistributed: https://stackoverflow.com/questions/42755820/how-to-use-return-sequences-option-and-timedistributed-layer-in-keras
    ## EXplanations of stateful parameter: http://philipperemy.github.io/keras-stateful-lstm/
    ## Good article on different modes of RNN (one-to-many, many-to-many etc.)

    branch = Sequential()
    branch.add(LSTM(c['lstm_shapes'][0], batch_input_shape=(c['batch_size'], c['timesteps'], c['features']),
                    return_sequences=c['return_sequence'], stateful=c['stateful'], name="lstm_1", go_backwards=c['go_backwards']))
    branch.add(LSTM(c['lstm_shapes'][0],
                    return_sequences=c['return_sequence'], name="lstm_2", stateful=c['stateful']))
    branch.add(LSTM(c['lstm_shapes'][0],
                    return_sequences=False, name="lstm_3", stateful=c['stateful']))
    branch.add(Dense(units=c['dense_shapes'][0], activation=c['activation'], name="dense_1"))
    branch.add(Dense(units=c['dense_shapes'][1], activation=c['activation'], name="dense_2"))

    return branch, c


def fundmentals_model_branch():
    """ branch to handle fundamentals data"""
    branch = Sequential()
    return branch


def nlp_branch_model():
    """branch to handle text and sentiment data"""
    branch = Sequential()
    return branch


def _resize_data_for_batches(data, batch_size):
    """this method is used to resize input data to fit selected batch size"""
    sample_size = data.shape[0]
    new_sample_size = math.floor(sample_size / batch_size) * batch_size

    return data[:new_sample_size]



def training_model(ticker, result_folder):
    """method to train the model"""

    # Initialize complex model.
    model, model_config = get_complex_model()
    model_config['epochs'] = 1
    model_config['mode'] = 'regression'
    ts_config = model_config['ts_config']

    # Retrieve data and then resize it to fit predefined batch size
    TS_X_train, TS_Y_train, TS_X_test, TS_Y_test = \
        get_ts_ticker_data(ticker, output_shape=
        (ts_config['batch_size'],ts_config['timesteps'],ts_config['features']),
                           classification=False)

    model.summary()
    print("Inputs: {}".format(model.input_shape))
    print("Outputs: {}".format(model.output_shape))

    # fit network
    # TODO: Not clear, how to sync up batch sizes across multiple branches

    loss = []
    val_loss = []


    for _ in range(model_config['epochs']):
        history = model.fit(TS_X_train, TS_Y_train, epochs=1, batch_size=model_config['ts_config']['batch_size'],
                        validation_data=(TS_X_test, TS_Y_test), verbose=2, shuffle=False)
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        model.reset_states()

#
    # # plot history
    pyplot.plot(loss, label='train')
    pyplot.plot(val_loss, label='test')
    pyplot.legend()
    pyplot.interactive(False)
    pyplot.savefig(result_folder+"loss_chart.png")
    pyplot.close()

    return model, model_config

def evaluate_model(model, X, Y, prediction_horizon, result_folder, classification=True):

    if classification:
        raise NotImplemented("Need to implement validation function for classification mode")

    time_series = list(range(X.shape[0]))

    Y_pred = model.predict(X, batch_size=batch_size)

    pyplot.plot(time_series, Y, label='real')
    pyplot.plot(time_series, Y_pred, label='predicted')
    pyplot.legend()
    pyplot.interactive(False)
    pyplot.savefig(result_folder+"true_vs_predicted.png")
    pyplot.close()



def get_sample_data(ticker, sample_size):

    file = TIMESERIES_FOLDER + "{0}_df.p".format(ticker)
    with open(file, "rb") as file:
        ticker_df = pickle.load(file)

    random_idx = random.randint(0, (ticker_df.shape[0] - sample_size - 1))

    X_sample = ticker_df.loc[random_idx:(random_idx + sample_size - 1), ['open', 'high', 'low', 'close', 'volume']]
    Y_sample = ticker_df.loc[(random_idx + 1):(random_idx+sample_size), ['close']]

    return X_sample, Y_sample

def _prepare_results_folder():
    result_folder = './results/{0}/'.format(str(uuid.uuid1()))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    return result_folder

def _dump_config(result_folder, config):
    with open(result_folder+'model_config.json', 'w') as outfile:
        json.dump(config, outfile)


batch_size = 64 # TODO: really bad practice. Global variable.
timesteps = 20

if __name__ == '__main__':
    result_folder = _prepare_results_folder()

    # Runtime configs
    ticker = 'FB'
    prediction_horizon = 20 # todo: as of now, this seems to be irrelevant feature.


    model, config = training_model(ticker, result_folder)

    TS_X_train, TS_Y_train, TS_X_test, TS_Y_test = \
        get_ts_ticker_data(ticker, output_shape=(batch_size, timesteps, 5), classification=False)

    evaluate_model(model, X=TS_X_test, Y=TS_Y_test,
                   classification=False, prediction_horizon=prediction_horizon, result_folder=result_folder)

    config['runtime'] = {'ticker' : ticker, 'prediction_horizon' : prediction_horizon} # adding some runtime parameters to config dump
    config['model'] = model.to_json() # adding model description to config dump
    _dump_config(result_folder, config)

    print(result_folder)

