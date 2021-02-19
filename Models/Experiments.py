import pandas as pd
import matplotlib.pyplot as plt
from ModelDataLoader import ModelDataLoader
from EvaluationFW import Evaluation
from SP import SP

class Experiments:

    def __init__(self, start_date, end_date, file_path_mp, nr_top_ch, train_prop, ratio_maj_min_class, use_time):
        self.data_loader = ModelDataLoader(start_date, end_date, file_path_mp, nr_top_ch, ratio_maj_min_class)
        self.SP_model = SP()
        self.idx_to_ch = self.data_loader.get_idx_to_ch_map()
        self.ch_to_idx = self.data_loader.get_ch_to_idx_map()
        self.clients_data_train = {}
        self.clients_data_test = {}
        self.attributions = {}
        self.train_prop = train_prop
        self.nr_top_ch = nr_top_ch

    def load_data(self):
        self.clients_data_train, self.clients_data_test = self.data_loader.get_clients_dict_split(self.train_prop)

        self.SP_model.load_train_test_data(self.clients_data_train, self.clients_data_test)

    def train_all(self):
        self.SP_model.train()

    def validate_pred(self):
        self.SP_model.validate()

    def load_attributions(self):
        self.attributions['SP'] = self.SP_model.get_normalized_attributions()

    def plot_attributions(self):
        channel_names = []
        for ch_idx in range(self.nr_top_ch):
            channel_names.append(self.idx_to_ch[ch_idx])

        df = pd.DataFrame({'Channel': channel_names, 'Attribution': self.attributions['SP']})
        ax = df.plot.bar(x='Channel', y='Attribution', rot=90)
        plt.tight_layout()
        plt.title('Attributions')
        plt.show()

    def profit_eval(self, total_budget):
        GA_df = self.data_loader.get_GA_df()
        converted_clients_df = self.data_loader.get_converted_clients_df()

        evaluation = Evaluation(GA_df, converted_clients_df, total_budget, self.attributions['SP'], self.ch_to_idx)
        evaluation.evaluate()


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-11.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=15, hour=23, minute=59, tz='UTC')

    train_proportion = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 6
    use_time = True
    total_budget = 1000

    experiments = Experiments(start_date, end_date, file_path_mp, nr_top_ch, train_proportion, ratio_maj_min_class, use_time)
    experiments.load_data()
    experiments.train_all()
    experiments.load_attributions()
    experiments.validate_pred()
    experiments.plot_attributions()
    experiments.profit_eval(total_budget)