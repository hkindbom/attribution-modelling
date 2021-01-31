from ChannelAttribution import auto_markov_model
import pandas as pd
import matplotlib.pyplot as plt

class MarkovModel:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.attributions = None
        self.transition_matrix = None
        self.removal_effects = None

    def train_all(self):
        auto_model_raw = auto_markov_model(self.dataframe,
                                  var_path='path',
                                  var_conv='total_conversions',
                                  var_value='total_conversion_value',
                                  var_null='total_null',
                                  out_more=True)
        self.attributions = auto_model_raw['result']
        self.transition_matrix = auto_model_raw['transition_matrix']
        self.removal_effects = auto_model_raw['removal_effects']


    def show_results(self):
        self.plot_attributions()
        self.plot_removal_effects()
        self.plot_transition_graph()

    def plot_attributions(self):
        ax = self.attributions.plot.bar('channel_name')
        plt.title('Markov Attributions')
        plt.show()

    def plot_transition_graph(self):
        print(self.transition_matrix)

    def plot_removal_effects(self):
        ax = self.removal_effects.plot.bar('channel_name')
        plt.title('Markov Removal Effects')
        plt.show()


def load_dataframe(data_file_path):
    dataframe = pd.read_csv(data_file_path, delimiter=';')
    if 'total_null' not in dataframe:
        dataframe['total_null'] = 0
    return dataframe

if __name__ == '__main__':
    data_file_path = '../Data/GA_data_processed.csv'#'../Data/sample-data-conversion-paths.csv'

    dataframe = load_dataframe(data_file_path)
    markov_model = MarkovModel(dataframe)
    markov_model.train_all()
    markov_model.show_results()

