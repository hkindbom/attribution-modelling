import tensorflow as tf
from tensorflow import keras
from attention import Attention
import numpy as np
import pandas as pd
from ModelDataLoader import ModelDataLoader

import sys
np.set_printoptions(threshold=sys.maxsize)

# https://stackabuse.com/solving-sequence-problems-with-lstm-in-keras/

class LSTM:

    def __init__(self, epochs, batch_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_len = None
        self.nr_features = None
        self.x_train = None
        self.y_train = None

    def setup_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(keras.layers.LSTM(units=32))
        #self.model.add(Attention())
        self.model.add(keras.layers.Dense(units=64, activation='relu'))
        self.model.add(keras.layers.Dense(units=1, activation='sigmoid'))

        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                           metrics=[keras.metrics.Recall(), keras.metrics.Precision(), 'accuracy'])

    def load_data(self, x_train, y_train):
        # sequences will be padded to the length of the longest individual sequence
        padded_data = keras.preprocessing.sequence.pad_sequences(x_train, value=-1, padding='post')
        self.max_seq_len = padded_data.shape[1]
        self.nr_features = len(np.unique(padded_data)) - 1
        self.x_train = self.one_hot_encode_x(padded_data)
        self.y_train = np.array(y_train)

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                 validation_split=0.2, verbose=1)

    def get_preds(self, x):
        padded_x = keras.preprocessing.sequence.pad_sequences(x, maxlen=self.max_seq_len, value=-1, padding='post')
        one_hot_x = self.one_hot_encode_x(padded_x)
        return self.model.predict(one_hot_x, verbose=0)

    def one_hot_encode_x(self, padded_data):
        return tf.one_hot(padded_data, self.nr_features, on_value=1, off_value=0, axis=-1)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-03-01.csv'
    start_date_data = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date_data = pd.Timestamp(year=2021, month=2, day=28, hour=23, minute=59, tz='UTC')

    start_date_cohort = pd.Timestamp(year=2021, month=2, day=5, hour=0, minute=0, tz='UTC')
    end_date_cohort = pd.Timestamp(year=2021, month=2, day=15, hour=23, minute=59, tz='UTC')

    nr_top_ch = 10
    ratio_maj_min_class = 1
    simulate = False
    cohort_size = 1000
    sim_time = 200

    data_loader = ModelDataLoader(start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mp,
                                 nr_top_ch, ratio_maj_min_class, simulate, cohort_size, sim_time)

    x_all, y_all = data_loader.get_all_seq_lists_and_labels()
    #x_all = [[0, 1, 1, 3, 1], [0, 2, 1]]
    #y_all = [1, 0]
    #print(x_all)
    #print(y_all)

    epochs = 20
    batch_size = 20
    learning_rate = 0.001

    lstm = LSTM(epochs, batch_size, learning_rate)
    lstm.load_data(x_all, y_all)
    lstm.setup_model()
    lstm.train()

    #print(np.c_[np.array(x_all), lstm.get_preds(x_all), np.array(y_all)])
    print('% conversions', sum(y_all)/len(y_all))
    print('nr samples: ', len(y_all))
