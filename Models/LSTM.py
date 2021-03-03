import tensorflow as tf
from tensorflow import keras
from attention import Attention
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
from ModelDataLoader import ModelDataLoader

import sys
np.set_printoptions(threshold=sys.maxsize)

class LSTM:

    def __init__(self, epochs, batch_size, learning_rate, validation_split=0.0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.max_seq_len = None
        self.nr_channels = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_train_raw = []

    def setup_model(self):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Masking(mask_value=0, input_shape=(self.x_train.shape[1], self.x_train.shape[2])))
        self.model.add(keras.layers.LSTM(units=32, return_sequences=True))
        self.model.add(Attention(name='attention_weight'))
        self.model.add(keras.layers.Dense(units=64, activation='relu'))
        self.model.add(keras.layers.Dense(units=1, activation='sigmoid'))

        self.model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                           metrics=[keras.metrics.Recall(), keras.metrics.Precision(), 'accuracy'])

    def load_data(self, x_train, y_train, x_test, y_test):
        self.x_train_raw = x_train
        # sequences will be padded to the length of the longest individual sequence
        padded_x_train = self.pad_x(x_train)
        self.max_seq_len = padded_x_train.shape[1]
        self.nr_channels = len(np.unique(padded_x_train)) - 1

        self.x_train = self.one_hot_encode_x(padded_x_train)
        self.y_train = np.array(y_train)

        padded_x_test = self.pad_x(x_test, self.max_seq_len)
        self.x_test = self.one_hot_encode_x(padded_x_test)
        self.y_test = np.array(y_test)

        self.setup_model()

    def train(self):
        history = self.model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size,
                                 validation_split=self.validation_split, verbose=1)

    def get_preds(self, one_hot_x):
        return self.model.predict(one_hot_x, verbose=0)

    def get_results(self):
        preds = np.round(self.get_preds(self.x_test))
        tn, fp, fn, tp = confusion_matrix(self.y_test, preds).ravel()
        auc = roc_auc_score(self.y_test, preds)

        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'auc': auc, 'logloss': np.nan}

    def pad_x(self, x, maxlen=None):
        return keras.preprocessing.sequence.pad_sequences(x, maxlen=maxlen, value=-1, padding='post')

    def one_hot_encode_x(self, padded_data):
        return tf.one_hot(padded_data, self.nr_channels, on_value=1, off_value=0, axis=-1)

    def get_normalized_attributions(self):
        unnorm_attr = self.get_non_normalized_attributions()
        norm_attr = [attribution / sum(unnorm_attr) for attribution in unnorm_attr]
        return norm_attr

    def get_non_normalized_attributions(self,):
        non_normalized_attributions = np.zeros(self.nr_channels)
        preds_w = self.get_preds(self.x_train)
        for ch_idx in range(self.nr_channels):
            x_train_w_o_ch, single_indices, ch_occur = self.remove_ch(self.x_train, ch_idx)
            preds_w_o = self.get_preds(x_train_w_o_ch)

            # correcting threshold for conversion/non-conversion
            preds_w_o[single_indices] = 0.5
            diff_preds = preds_w - preds_w_o
            non_normalized_attributions[ch_idx] = max(diff_preds.sum(), 0) / ch_occur
        return non_normalized_attributions

    def remove_ch(self, x, ch_idx):
        x = x.numpy()
        single_indices = []
        ch_occur = 0
        for seq_idx, seq in enumerate(x):
            t_p_to_del = []
            for t_p_idx, t_p in enumerate(seq):
                if np.argmax(t_p) == ch_idx:
                    t_p_to_del.append(t_p_idx)
                    ch_occur += 1

            seq_deleted = np.delete(x[seq_idx], t_p_to_del, 0)
            new_seq = np.concatenate((seq_deleted, np.zeros((len(t_p_to_del), self.nr_channels))), axis=0)
            x[seq_idx] = new_seq

            # identifying sequences with only one channel
            if new_seq.max() == 0:
                single_indices.append(seq_idx)
        return tf.constant(x), single_indices, ch_occur

    def get_attention_weights(self):
        layer_name = 'attention_weight'
        attention_layer_model = keras.Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
        attention_output = attention_layer_model(self.x_train)
        return attention_output

    def get_touchpoint_attr(self, seq_len):
        attention_weights = self.get_attention_weights().numpy()
        touchpoint_attr = np.zeros(seq_len)
        nr_matched_seq = 0
        for seq_idx, seq_attention in enumerate(attention_weights):
            if self.y_train[seq_idx] == 1 and len(self.x_train_raw[seq_idx]) == seq_len:
                touchpoint_attr += seq_attention[:seq_len]
                nr_matched_seq += 1
        print('Basing touchpoint attribution on ', nr_matched_seq, ' sequences')
        return touchpoint_attr / touchpoint_attr.sum()

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
    simulate = True
    cohort_size = 1000
    sim_time = 200

    data_loader = ModelDataLoader(start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mp,
                                 nr_top_ch, ratio_maj_min_class, simulate, cohort_size, sim_time)

    print('Theoretical max accuracy on all data is: ', data_loader.get_theo_max_accuracy())

    x_train, y_train, x_test, y_test = data_loader.get_seq_lists_split(train_prop)

    epochs = 10
    batch_size = 20
    learning_rate = 0.001
    validation_split = 0.2
    tp_attr_seq_len = 5

    lstm = LSTM(epochs, batch_size, learning_rate, validation_split)
    lstm.load_data(x_train, y_train, x_test, y_test)
    lstm.train()
    print('Channel attributions: ', lstm.get_normalized_attributions())
    print('Touchpoint attributions: ', lstm.get_touchpoint_attr(tp_attr_seq_len))
    print(lstm.get_results())

