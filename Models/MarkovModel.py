from ChannelAttribution import markov_model
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

class MarkovModel:
    def __init__(self, dataframe, order, show_top_results=False):
        self.dataframe = dataframe
        self.attributions = None
        self.transition_matrix = None
        self.removal_effects = None
        self.order = order
        self.show_top_results = show_top_results

    def train_all(self):

        model_raw = markov_model(self.dataframe,
                                  var_path='path',
                                  var_conv='total_conversions',
                                  var_value='total_conversion_value',
                                  var_null='total_null',
                                  out_more=True,
                                  order=self.order)
        if self.show_top_results:
            nr_results = 15
        else:
            nr_results = -1

        self.attributions = model_raw['result'].sort_values(by=['total_conversion_value'], ascending=False)[:nr_results]
        self.transition_matrix = model_raw['transition_matrix'][:nr_results]
        self.removal_effects = model_raw['removal_effects'].sort_values(by=['removal_effects_conversion_value'], ascending=False)[:nr_results]

    def show_results(self):
        self.plot_attributions()
        self.plot_removal_effects()
        self.plot_transition_graph()

    def plot_attributions(self):
        ax = self.attributions.plot.bar('channel_name')
        plt.title('Markov, order ' + str(self.order))
        plt.ylabel('Absolute attributed conversions')
        ax.legend(['Nr conversions', 'Conversion value'])
        plt.show()

    def plot_transition_graph(self):
        print(self.transition_matrix)

    def plot_removal_effects(self):
        ax = self.removal_effects.plot.bar('channel_name')
        plt.title('Markov, order ' + str(self.order))
        plt.ylabel('Normalized attributions')
        plt.xlabel('Channel')
        ax.legend(['Nr conversions', 'Conversion value'])
        plt.tight_layout()
        plt.savefig('../Plots/Markov_model_order_' + str(self.order) + '_' + str(datetime.now()).replace(' ','_').replace(':','_') + '.png', dpi=200)
        plt.show()



def load_dataframe(data_file_path):
    dataframe = pd.read_csv(data_file_path, delimiter=';')
    print(dataframe.dtypes)
    if 'total_null' not in dataframe:
        dataframe['total_null'] = 0
    return dataframe

if __name__ == '__main__':
    data_file_path = '../Data/channel_journey_data_processed.csv'
    order = 3
    show_top_results = True
    dataframe = load_dataframe(data_file_path)
    markovmodel = MarkovModel(dataframe, order, show_top_results)
    markovmodel.train_all()
    markovmodel.show_results()

