import pickle
import quandl
import pandas as pd
from pandas import DataFrame
from keras.utils import to_categorical
import pandas.api.types as ptypes
from sklearn.preprocessing import MinMaxScaler


# Quandl configuration
quandl.ApiConfig.api_key = 'h6tfVg1ps54v4bcpc3xz'

# Data source configuration
DATA_FOLDER = 'data/'
TIMESERIES_FOLDER = "timeseries/"
SP500_FILE= DATA_FOLDER+"sp500.p"

def timeseries_data_pipeline():

    #1. load scrapped list of SP500 companies
    with open(SP500_FILE,'rb') as file:
        sp500_df = pickle.load(file)

    #2. For each ticker from SP500 load it's timeseries data
    sp500_tickers = sp500_df.ticker

    # TODO: delay one hot encoding. need to decide where it's more appropriate to do.
    # #3. Prepare one-hot encoded sector id for each ticker
    # encoded_sector_id = to_categorical(sp500_df.sector_id)

    for index, ticker in enumerate(sp500_tickers):
        print("Handling ticker {0}".format(ticker))
        ticker_df = quandl.get_table('WIKI/PRICES', ticker=ticker)

        if ticker_df.shape[0] == 0: # to handle cases when Quandle doesn't return data for a given ticker.
            print("Cannot retrieve Quandl data for {0}. Skipping...".format(ticker))
            continue

        scaler = MinMaxScaler()
        numeric_columns = ["open","high","low","close","volume","ex-dividend","split_ratio",
                               "adj_open","adj_high","adj_low","adj_close","adj_volume"]

        ticker_df[numeric_columns] = scaler.fit_transform(ticker_df[numeric_columns])

        # 4. Augment time series data with industry sector information
        ticker_df['sector_id'] = sp500_df.sector_id.loc[index]

        # 5. Save to the disk as data frame
        ticker_df_file = DATA_FOLDER+TIMESERIES_FOLDER+"{0}_df.p".format(ticker)
        with open(ticker_df_file,"wb") as file:
            pickle.dump(ticker_df, file)

    return None


def nlp_data_pipeline():
    pass

def fundamentals_Data_pipeline():
    pass


if __name__ == '__main__':
    timeseries_data_pipeline()
