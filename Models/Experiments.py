import pandas as pd
import matplotlib.pyplot as plt
from ModelDataLoader import ModelDataLoader
from EvaluationFW import Evaluation
from SP import SP
from LTA import LTA
from LR import LR

class Experiments:

    def __init__(self, start_date, end_date, file_path_mp, nr_top_ch, train_prop, ratio_maj_min_class, use_time):
        self.data_loader = ModelDataLoader(start_date, end_date, file_path_mp, nr_top_ch, ratio_maj_min_class)
        self.use_time = use_time
        self.SP_model = SP()
        self.LTA_model = LTA()
        self.LR_model = LR()
        self.idx_to_ch = self.data_loader.get_idx_to_ch_map()
        self.ch_to_idx = self.data_loader.get_ch_to_idx_map()
        self.clients_data_train = {}
        self.clients_data_test = {}
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

        self.attributions = {}
        self.train_prop = train_prop
        self.nr_top_ch = nr_top_ch

    def load_data(self):
        self.clients_data_train, self.clients_data_test = self.data_loader.get_clients_dict_split(self.train_prop)
        self.x_train, self.y_train, self.x_test, self.y_test = self.data_loader.get_feature_matrix_split(self.train_prop, self.use_time)

        self.SP_model.load_train_test_data(self.clients_data_train, self.clients_data_test)
        self.LTA_model.load_train_test_data(self.clients_data_train, self.clients_data_test)
        self.LR_model.load_train_test_data(self.x_train, self.y_train, self.x_test, self.y_test)

    def train_all(self):
        self.SP_model.train()
        self.LTA_model.train()
        self.LR_model.train()

    def validate_pred(self):
        print('SP performance')
        self.SP_model.validate()
        print('LTA performance')
        self.LTA_model.validate()
        print('LR performance')
        self.LR_model.validate()

    def load_attributions(self):
        self.attributions['SP'] = self.SP_model.get_normalized_attributions()
        self.attributions['LTA'] = self.LTA_model.get_normalized_attributions()
        self.attributions['LR'] = self.LR_model.get_normalized_attributions()

    def plot_attributions(self):
        channel_names = []
        for ch_idx in range(self.nr_top_ch):
            channel_names.append(self.idx_to_ch[ch_idx])

        df = pd.DataFrame({'Channel': channel_names,
                           'LTA Attribution': self.attributions['LTA'],
                           'SP Attribution': self.attributions['SP'],
                           'LR Attribution': self.attributions['LR']})

        ax = df.plot.bar(x='Channel', rot=90)
        ax.set_xlabel("Source / Medium")
        plt.tight_layout()
        plt.title('Attributions')
        plt.show()

    def profit_eval(self, total_budget):
        GA_df = self.data_loader.get_GA_df()
        converted_clients_df = self.data_loader.get_converted_clients_df()

        for model_name in self.attributions:
            print(model_name)
            evaluation = Evaluation(GA_df, converted_clients_df, total_budget, self.attributions[model_name], self.ch_to_idx)
            evaluation.evaluate()


if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-17.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=15, hour=23, minute=59, tz='UTC')

    train_proportion = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 2
    use_time = True
    total_budget = 1000

    experiments = Experiments(start_date, end_date, file_path_mp, nr_top_ch, train_proportion, ratio_maj_min_class, use_time)
    experiments.load_data()
    experiments.train_all()
    experiments.load_attributions()
    experiments.validate_pred()
    experiments.plot_attributions()
    experiments.profit_eval(total_budget)