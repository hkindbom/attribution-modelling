from ModelDataLoader import ModelDataLoader
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

# Code inspired by SP.py in https://github.com/rk2900/deep-conv-attr

class SP:
    def __init__(self, start_date, end_date, file_path_mp, nr_top_ch,
                 train_prop=0.8, ratio_maj_min_class=1):

        self.data_loader = ModelDataLoader(start_date, end_date, file_path_mp, nr_top_ch, ratio_maj_min_class)
        self.clients_data_train = {}
        self.clients_data_test = {}
        self.idx_to_ch = self.data_loader.get_idx_to_ch_map()
        self.channel_value = {}
        self.channel_time = {}
        self.prob = {}
        self.train_prop = train_prop

    def load_train_test_data(self):
        self.clients_data_train, self.clients_data_test = self.data_loader.get_clients_dict_split(self.train_prop)

    def train(self):
        self.load_train_test_data()
        for client_id in self.clients_data_train:
            self.add_client_to_model(client_id)
        self.calc_prob()

    def add_client_to_model(self, client_id):
        client_label = self.clients_data_train[client_id]['label']
        for channel_idx in self.clients_data_train[client_id]['session_channels']:
            if channel_idx in self.channel_value:
                self.channel_value[channel_idx] += client_label
                self.channel_time[channel_idx] += 1.
            else:
                self.channel_value[channel_idx] = client_label
                self.channel_time[channel_idx] = 1.

    def calc_prob(self):
        for channel_idx in self.channel_value:
            self.prob[channel_idx] = self.channel_value[channel_idx] / self.channel_time[channel_idx]

    def get_prediction(self, client_id, clients_data):
        pred = 1
        for channel_idx in clients_data[client_id]['session_channels']:
            mult_factor = (1 - self.prob[channel_idx]) if channel_idx in self.prob else 1
            pred *= mult_factor
        return round(1 - pred)

    def validate(self):
        labels = []
        preds = []
        for client_id in self.clients_data_test:
            preds.append(self.get_prediction(client_id, self.clients_data_test))
            labels.append(self.clients_data_test[client_id]['label'])

        self.show_performance(labels, preds)

    def show_performance(self, labels, preds):
        auc = roc_auc_score(labels, preds)
        logloss = log_loss(labels, preds)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        print('Accuracy ', (tn+tp)/(tn+tp+fp+fn))
        print('AUC: ', auc)
        print('Log-loss: ', logloss)
        print('tn:', tn, ' fp:', fp, ' fn:', fn, ' tp:', tp)
        print('precision: ', tp / (tp + fp), ' ability of the classifier not to label as positive a sample that is negative')
        print('recall: ', tp / (tp + fn), ' ability of the classifier to find all the positive samples')

    def plot_attributions(self):
        channel_names = []
        channel_attribution = self.get_non_normalized_attributions()

        for channel_idx, _ in enumerate(self.prob):
            channel_names.append(self.idx_to_ch[channel_idx])

        df = pd.DataFrame({'Channel': channel_names, 'Attribution': channel_attribution})
        ax = df.plot.bar(x='Channel', y='Attribution', rot=90)
        plt.tight_layout()
        plt.title('Attributions - SP model')
        plt.show()

    def get_non_normalized_attributions(self):
        unnorm_attr = np.zeros(len(self.prob))
        for ch_idx, _ in enumerate(unnorm_attr):
            unnorm_attr[ch_idx] = self.prob[ch_idx]
        return unnorm_attr.tolist()

    def get_normalized_attributions(self):
        unnorm_attr = self.get_non_normalized_attributions()
        norm_attr = [attribution / sum(unnorm_attr) for attribution in unnorm_attr]
        return norm_attr

    def get_GA_df(self):
        return self.data_loader.get_GA_df()

    def get_converted_clients_df(self):
        return self.data_loader.get_converted_clients_df()

    def get_ch_to_idx_map(self):
        return self.data_loader.get_ch_to_idx_map()

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-17.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=10, hour=23, minute=59, tz='UTC')

    train_proportion = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 8

    SP_model = SP(start_date, end_date, file_path_mp, nr_top_ch, train_proportion, ratio_maj_min_class)
    SP_model.train()
    SP_model.validate()
    SP_model.plot_attributions()