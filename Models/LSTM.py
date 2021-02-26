import tensorflow as tf
from tensorflow import keras
from attention import Attention
import numpy as np
import pandas as pd
from ModelDataLoader import ModelDataLoader

#LSTM
#Attention
#Transfer learning

class LSTM:

    def __init__(self):
        self.epochs = 1
        self.x_train = None
        self.y_train = None

    def setup_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(keras.layers.LSTM(units=64, input_shape=(self.x_train.shape[1], 1)))
        #self.model.add(Attention())
        self.model.add(keras.layers.Dense(units=64, activation='relu'))
        self.model.add(keras.layers.Dense(units=1))

        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.Adam(1e-4),
                           metrics=['accuracy'])

    def encode_data(self, x_train, y_train):
        # sequences will be padded to the length of the longest individual sequence
        padded_data = keras.preprocessing.sequence.pad_sequences(x_train)
        print(padded_data.reshape(padded_data.shape[0], padded_data.shape[1], 1))
        self.x_train = padded_data.reshape(padded_data.shape[0], padded_data.shape[1], 1)
        self.y_train = np.array(y_train)
        """
        data = tf.constant(padded_data)
        indexer = keras.layers.experimental.preprocessing.IntegerLookup()#mask_value=None)
        indexer.adapt(data)
        encoder = keras.layers.experimental.preprocessing.CategoryEncoding(output_mode='binary')
        encoder.adapt(indexer(data))

        print(encoder(indexer(data)))
        print(encoder(indexer(tf.constant([3, 2, 10]))))
        print(indexer.get_vocabulary())
        """

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=100, validation_split=0.2, verbose=1)

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-22.csv'
    start_date_data = pd.Timestamp(year=2021, month=2, day=2, hour=0, minute=0, tz='UTC')
    end_date_data = pd.Timestamp(year=2021, month=2, day=21, hour=23, minute=59, tz='UTC')

    start_date_cohort = pd.Timestamp(year=2021, month=2, day=5, hour=0, minute=0, tz='UTC')
    end_date_cohort = pd.Timestamp(year=2021, month=2, day=13, hour=23, minute=59, tz='UTC')

    train_proportion = 0.8
    nr_top_ch = 10
    ratio_maj_min_class = 1

    data_loader = ModelDataLoader(start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mp,
                                 nr_top_ch, ratio_maj_min_class)
    x_all, y_all = data_loader.get_all_seq_lists_and_labels()

    data = [[2, 3, 4, 7], [2, 5]]
    y_train = np.array([1, 0])

    lstm = LSTM()
    lstm.encode_data(x_all, y_all)
    lstm.setup_model()
    lstm.train()
