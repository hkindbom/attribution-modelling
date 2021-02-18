from ModelDataLoader import ModelDataLoader
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
        GA_df = self.data_loader.get_GA_df()
        GA_df.to_csv('hejsan.csv')

    def train(self):
        self.load_train_test_data()
        client_ids_train = list(self.clients_data_train.keys())

        for client_id in client_ids_train:
            self.add_client_to_model(client_id)
        self.calc_prob()

    def add_client_to_model(self, client_id):
        client_label = self.clients_data_train[client_id]['label']
        last_channel = self.clients_data_train[client_id]['session_channels'][-1]
        if last_channel in self.channel_value:
            self.channel_value[last_channel] += client_label
            self.channel_time[last_channel] += 1.
        else:
            self.channel_value[last_channel] = client_label
            self.channel_time[last_channel] = 1.

    def calc_prob(self):
        for channel_idx in self.channel_value:
            self.prob[channel_idx] = self.channel_value[channel_idx] / self.channel_time[channel_idx]

    def get_attributions(self):
        channel_attributions = []
        for channel_idx in self.prob.keys():
            channel_attributions.append(self.prob[channel_idx])
        channel_attributions = [attribution/sum(channel_attributions) for attribution in channel_attributions]
        return channel_attributions

    def plot_attributions(self):
        channel_names = []
        channel_attribution = []

        for channel_idx in self.prob.keys():
            channel_names.append(self.idx_to_ch[channel_idx])
            channel_attribution.append(self.prob[channel_idx])

        df = pd.DataFrame({'Channel': channel_names, 'Attribution': channel_attribution})
        ax = df.plot.bar(x='Channel', y='Attribution', rot=90)
        plt.tight_layout()
        plt.title('Attributions - LTA model')
        plt.show()


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-17.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=10, hour=23, minute=59, tz='UTC')

    train_proportion = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 1

    LTA_model = LTA(start_date, end_date, file_path_mp, nr_top_ch, train_proportion, ratio_maj_min_class)
    LTA_model.train()
    attributions = LTA_model.get_attributions()
    LTA_model.plot_attributions()