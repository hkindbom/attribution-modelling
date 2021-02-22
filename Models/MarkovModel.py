from ChannelAttribution import markov_model
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from DataProcessing import DataProcessing

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
        self.transition_matrix = model_raw['transition_matrix']
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

if __name__ == '__main__':
    file_path_GA_aggregated = '../Data/Analytics_raw_data_sample.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=1, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=15, hour=23, minute=59, tz='UTC')

    data_processing = DataProcessing(start_date, end_date, file_path_GA_aggregated = file_path_GA_aggregated)
    data_processing.process_aggregated_data()
    dataframe = data_processing.get_GA_aggr_df()
    
    order = 3
    show_top_results = True
    markovmodel = MarkovModel(dataframe, order, show_top_results)
    markovmodel.train_all()
    markovmodel.show_results()

