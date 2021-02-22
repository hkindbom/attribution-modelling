from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from ModelDataLoader import ModelDataLoader

class LTA:
    def __init__(self):
        self.clients_data_train = {}
        self.clients_data_test = {}
        self.channel_value = {}
        self.channel_time = {}
        self.prob = {}

    def load_train_test_data(self, clients_data_train, clients_data_test):
        self.clients_data_train, self.clients_data_test = clients_data_train, clients_data_test

    def train(self):
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

    def get_non_normalized_attributions(self):
        unnorm_attr = []
        for ch_idx in range(len(self.prob)):
            unnorm_attr.append(self.prob[ch_idx])
        return unnorm_attr

    def get_normalized_attributions(self):
        unnorm_attr = self.get_non_normalized_attributions()
        norm_attr = [attribution / sum(unnorm_attr) for attribution in unnorm_attr]
        return norm_attr

    def plot_attributions(self, idx_to_ch):
        channel_attribution = self.get_non_normalized_attributions()
        channel_names = []
        for ch_idx in range(len(self.prob)):
            channel_names.append(idx_to_ch[ch_idx])

        df = pd.DataFrame({'Channel': channel_names, 'Attribution': channel_attribution})
        ax = df.plot.bar(x='Channel', y='Attribution', rot=90)
        plt.tight_layout()
        plt.title('Attributions - LTA model')
        plt.show()

    def get_prediction(self, client_id, clients_data):
        channel_idx = clients_data[client_id]['session_channels'][-1]
        pred = self.prob[channel_idx]
        return round(pred)

    def get_results(self):
        labels, preds = self.get_labels_and_preds()
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        auc = roc_auc_score(labels, preds)
        logloss = log_loss(labels, preds)

        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'auc': auc, 'logloss': logloss}

    def get_labels_and_preds(self):
        labels = []
        preds = []
        client_ids_test = list(self.clients_data_test)
        for client_id in client_ids_test:
            preds.append(self.get_prediction(client_id, self.clients_data_test))
            labels.append(self.clients_data_test[client_id]['label'])
        return labels, preds

    def validate(self):
        res = self.get_results()
        print('Accuracy ', (res['tn']+res['tp'])/(res['tn']+res['tp']+res['fp']+res['fn']))
        print('AUC: ', res['auc'])
        print('Log-loss: ', res['logloss'])
        print('tn:', res['tn'], ' fp:', res['fp'], ' fn:', res['fn'], ' tp:', res['tp'])
        print('precision: ', res['tp'] / (res['tp'] + res['fp']))
        print('recall: ', res['tp'] / (res['tp'] + res['fn']))


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-17.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=15, hour=23, minute=59, tz='UTC')

    train_proportion = 0.8
    nr_top_ch = 10
    ratio_maj_min_class = 2

    data_loader = ModelDataLoader(start_date, end_date, file_path_mp, nr_top_ch, ratio_maj_min_class)
    clients_data_train, clients_data_test = data_loader.get_clients_dict_split(train_proportion)

    LTA_model = LTA()
    LTA_model.load_train_test_data(clients_data_train, clients_data_test)
    LTA_model.train()
    LTA_model.validate()
    LTA_model.plot_attributions(data_loader.get_idx_to_ch_map())
