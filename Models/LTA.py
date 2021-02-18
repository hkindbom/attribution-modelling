from ModelDataLoader import ModelDataLoader
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd


class LTA:
    def __init__(self, start_date, end_date, file_path_mp, nr_top_ch, train_prop=0.8, ratio_maj_min_class=1):
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
        last_channel_in_path = self.clients_data_train[client_id]['session_channels'][-1]
        if last_channel_in_path in self.channel_value:
            self.channel_value[last_channel_in_path] += client_label
            self.channel_time[last_channel_in_path] += 1.
        else:
            self.channel_value[last_channel_in_path] = client_label
            self.channel_time[last_channel_in_path] = 1.

    def calc_prob(self):
        for channel_idx in self.channel_value:
            self.prob[channel_idx] = self.channel_value[channel_idx] / self.channel_time[channel_idx]

    def get_attributions(self):
        channel_attributions = []
        for channel_idx in self.prob:
            channel_attributions.append(self.prob[channel_idx])
        channel_attributions = [attribution/sum(channel_attributions) for attribution in channel_attributions]
        return channel_attributions

    def plot_attributions(self):
        channel_names = []
        channel_attribution = []

        for channel_idx in self.prob:
            channel_names.append(self.idx_to_ch[channel_idx])
            channel_attribution.append(self.prob[channel_idx])

        df = pd.DataFrame({'Channel': channel_names, 'Attribution': channel_attribution})
        ax = df.plot.bar(x='Channel', y='Attribution', rot=90)
        plt.tight_layout()
        plt.title('Attributions - LTA model')
        plt.show()

    def get_prediction(self, client_id, clients_data):
        channel_idx = clients_data[client_id]['session_channels'][-1]
        pred = self.prob[channel_idx]
        return round(pred)

    def validate(self):
        labels = []
        preds = []
        client_ids_test = list(self.clients_data_test)
        for client_id in client_ids_test:
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
        print('precision: ', tp / (tp + fp))
        print('recall: ', tp / (tp + fn))

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
    end_date = pd.Timestamp(year=2021, month=2, day=15, hour=23, minute=59, tz='UTC')

    train_proportion = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 1

    LTA_model = LTA(start_date, end_date, file_path_mp, nr_top_ch, train_proportion, ratio_maj_min_class)
    LTA_model.train()
    LTA_model.validate()
    LTA_model.plot_attributions()
