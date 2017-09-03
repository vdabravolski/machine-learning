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

TIMESERIES_FOLDER = "data/timeseries/"


def get_ts_ticker_data(ticker, input_dim, train_test_ratio=0.7, sliding_window=True):
    """method to do necessary data massage"""

    file = TIMESERIES_FOLDER + "{0}_df.p".format(ticker)
    with open(file, "rb") as file:
        ticker_df = pickle.load(file)

    # explicitly define input dimensions
    batch_size = input_dim[0]
    timesteps = input_dim[1]
    features = input_dim[2]

    augmented_df = pd.DataFrame(columns=ticker_df.columns)

    if sliding_window:  # implementation of sliding window. Each window is moving by 1 element.
        for idx in range(ticker_df.shape[0] - timesteps):
            augmented_df = augmented_df.append(ticker_df[idx: idx + timesteps])
            # for idx_ts in range(timesteps):
            #     combined_idx = idx_df+idx_ts

    augmented_df.shape

    # Assume that we use "close" price as an indicator whether the stock is up and down.
    Y = augmented_df.close[:-1].reset_index(drop=True) < augmented_df.close[1:].reset_index(drop=True)
    Y = Y.astype(int)
    print(augmented_df.columns)
    augmented_df = augmented_df.reset_index(drop=True)
    X = augmented_df.loc[:(augmented_df.shape[0] - 2),
        ['open', 'high', 'low', 'close', 'volume']]  # TODO: reduced number of features is used.

    # TODO: right now sector_id is not one-hot encoded.
    train_index = int(X.shape[0] * train_test_ratio)

    # TODO: P0 with sliding window approach, labels are off on the boundaries.
    Y_train = Y.values[:train_index]
    # Y_train = to_categorical(Y_train, num_classes=2)

    Y_test = Y.values[(train_index + 1):]
    # Y_test = to_categorical(Y_test, num_classes=2)

    X_train = X.values[:train_index, :]
    X_test = X.values[(train_index + 1):, :]

    # TODO: when resizing, we are trimming a lot of data to fit into the batch*timestep dimensions.
    X_train = _resize_data_for_batches(X_train, batch_size * timesteps)
    X_test = _resize_data_for_batches(X_test, batch_size * timesteps)
    Y_train = _resize_data_for_batches(Y_train, batch_size * timesteps)
    Y_test = _resize_data_for_batches(Y_test, batch_size * timesteps)

    # resize to batches
    X_train = np.reshape(X_train, (-1, timesteps, features))
    X_test = np.reshape(X_test, (-1, timesteps, features))
    Y_train = np.reshape(Y_train, (-1, timesteps))
    Y_test = np.reshape(Y_test, (-1, timesteps))

    print("Input data shape: \n X train: {0}, Y train: {1} \n X test: {2}, Y test: {3}".
          format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
    return X_train, Y_train, X_test, Y_test


def get_complex_model():
    """a model which combine several branches with various inputs"""

    # A dictionary with specific config. Will be used to track experiments settings.
    model_config = {'name': 'complex_model'}
    model_config['optimizer'] = 'adam'
    model_config['loss'] = 'binary_crossentropy'

    # TODO: in future add here nlp and fundamentals branches
    model, ts_config = ts_model_branch()
    model_config['ts_config'] = ts_config
    model.compile(loss=model_config['loss'], optimizer=model_config['optimizer'], metrics=['accuracy'])

    return model, model_config

def ts_model_branch():
    c = {'name': 'ts_branch'}
    c['lstm2_shape'] = [100]  # TODO: #2 element is inline with with batch size.
    c['activation'] = 'softmax'
    c['lstm_units'] = 5
    c['batch_size'] = 128
    c['timesteps'] = 50
    c['return_sequence'] = False
    c['stateful'] = True
    c['features'] = 5  # TODO: this is a hardcoded value. Need to come up with not hardcoded alternative. Thi defines input dimension.
    c['output'] = 1

    # # TODO: likely, this will work only for timesteps=1. Need to rethink assignment of predicted values given the timestep.
    ## Explanations of return_sequence and TimeDistributed: https://stackoverflow.com/questions/42755820/how-to-use-return-sequences-option-and-timedistributed-layer-in-keras
    ## EXplanations of stateful parameter: http://philipperemy.github.io/keras-stateful-lstm/
    branch = Sequential()
    branch.add(LSTM(c['lstm_units'], batch_input_shape=(c['batch_size'], c['timesteps'], c['features']),
                    return_sequences=True, stateful=c['stateful'], name="lstm_1"))
    branch.add(Dropout(0.2, name="dropout_1"))
    branch.add(LSTM(c['lstm_units'],
                    return_sequences=False, name="lstm_2", stateful=c['stateful']))
    branch.add(Dropout(0.2, name="dropout_2"))
    branch.add(Dense(units=c['output'], activation=c['activation'],
               name="dense_1"))
    # branch.add(Activation(c['activation'], name="activation_test_1"))

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


def training_model(ticker):
    """method to train the model"""

    # Initialize complex model.
    model, model_config = get_complex_model()
    model_config['epochs'] = 10
    ts_config = model_config['ts_config']

    print("TS branch config".format(ts_config))

    # Retrieve data and then resize it to fit predefined batch size
    TS_X_train, TS_Y_train, TS_X_test, TS_Y_test = get_ts_ticker_data(ticker, input_dim=(ts_config['batch_size'],
                                                                                         ts_config['timesteps'],
                                                                                         ts_config['features']))

    model.summary()
    print("Inputs: {}".format(model.input_shape))
    print("Outputs: {}".format(model.output_shape))

    # fit network
    # TODO: Not clear, how to sync up batch sizes across multiple branches
    history = model.fit(TS_X_train, TS_Y_train, epochs=model_config['epochs'],
                        batch_size=model_config['ts_config']['batch_size'],
                        validation_data=(TS_X_test, TS_Y_test), verbose=2, shuffle=False)

    # plot history
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.interactive(False)
    pyplot.show()

    print(model_config)


if __name__ == '__main__':
    ticker = 'FB'

    training_model(ticker)
