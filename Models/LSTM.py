import tensorflow as tf
from tensorflow import keras
from attention import Attention
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
from ModelDataLoader import ModelDataLoader

import sys
np.set_printoptions(threshold=sys.maxsize)

class LSTM:

    def __init__(self, epochs, batch_size, learning_rate):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_seq_len = None
        self.nr_features = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

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

    def load_data(self, x_train, y_train, x_test, y_test):
        # sequences will be padded to the length of the longest individual sequence
        padded_x_train = self.pad_x(x_train)
        self.max_seq_len = padded_x_train.shape[1]
        self.nr_features = len(np.unique(padded_x_train)) - 1

        self.x_train = self.one_hot_encode_x(padded_x_train)
        self.y_train = np.array(y_train)

        padded_x_test = self.pad_x(x_test, self.max_seq_len)
        self.x_test = self.one_hot_encode_x(padded_x_test)
        self.y_test = np.array(y_test)

        self.setup_model()

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                 validation_split=0.2, verbose=1)

    def get_preds(self, one_hot_x):
        return self.model.predict(one_hot_x, verbose=0)

    def get_results(self):
        preds = np.round(self.get_preds(self.x_test))
        tn, fp, fn, tp = confusion_matrix(self.y_test, preds).ravel()
        auc = roc_auc_score(self.y_test, preds)
        logloss = log_loss(self.y_test, preds)

        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'auc': auc, 'logloss': logloss}

    def pad_x(self, x, maxlen=None):
        return keras.preprocessing.sequence.pad_sequences(x, maxlen=maxlen, value=-1, padding='post')

    def one_hot_encode_x(self, padded_data):
        return tf.one_hot(padded_data, self.nr_features, on_value=1, off_value=0, axis=-1)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-03-01.csv'
    start_date_data = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date_data = pd.Timestamp(year=2021, month=2, day=21, hour=23, minute=59, tz='UTC')

    start_date_cohort = pd.Timestamp(year=2021, month=2, day=5, hour=0, minute=0, tz='UTC')
    end_date_cohort = pd.Timestamp(year=2021, month=2, day=13, hour=23, minute=59, tz='UTC')

    nr_top_ch = 10
    ratio_maj_min_class = 1
    train_prop = 0.8
    simulate = False
    cohort_size = 1000
    sim_time = 200

    data_loader = ModelDataLoader(start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mp,
                                 nr_top_ch, ratio_maj_min_class, simulate, cohort_size, sim_time)

    print('Theoretical max accuracy on all data is: ', data_loader.get_theo_max_accuracy())

    x_train, y_train, x_test, y_test = data_loader.get_seq_lists_split(train_prop)

    epochs = 20
    batch_size = 20
    learning_rate = 0.001

    lstm = LSTM(epochs, batch_size, learning_rate)
    lstm.load_data(x_train, y_train, x_test, y_test)
    lstm.train()
    print(lstm.get_results())

