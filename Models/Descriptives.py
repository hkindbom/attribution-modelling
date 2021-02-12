import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from DataProcessing import DataProcessing


class Descriptives:
    def __init__(self, start_date, end_date, file_path_mp=None):
        self.start_date = start_date
        self.end_date = end_date
        self.data_processing = DataProcessing(self.start_date, self.end_date, file_path_mp)
        self.GA_df = None
        self.MP_df = None
        self.converted_clients_df = None
        self.funnel_df = None
        self.read_data()

    def read_data(self):
        self.data_processing.process_all()
        self.GA_df = self.data_processing.get_GA_df()
        self.MP_df = self.data_processing.get_MP_df()
        self.converted_clients_df = self.data_processing.get_converted_clients_df()
        self.funnel_df = self.data_processing.get_funnel_df()

    def get_conversion_paths(self):
        return self.GA_df.loc[self.GA_df['converted_eventually'] == 1]

    def get_non_conversion_paths(self):
        return self.GA_df.loc[self.GA_df['converted_eventually'] == 0]

    def get_conversion_paths_last(self):
        conversion_paths = self.get_conversion_paths()
        return conversion_paths.loc[conversion_paths['conversion'] == 1]

    def get_conversion_paths_not_last(self):
        conversion_paths = self.get_conversion_paths()
        return conversion_paths.loc[conversion_paths['conversion'] == 0]

    def plot_path_length_GA(self):
        conversion_paths = self.get_conversion_paths()
        path_lengths = []
        for client, path in conversion_paths.groupby(level=0):
            path_lengths.append(len(path))
        temp_df = pd.DataFrame({'freq': path_lengths})
        temp_df.groupby('freq', as_index=False).size().plot(x='freq', y='size', kind='bar', legend=False)
        plt.title('Conversion path lengths')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.show()

    def plot_path_duration_GA(self, nr_bars=10):
        conversion_paths = self.get_conversion_paths()
        path_duration = []
        for client, path in conversion_paths.groupby(level=0):
            path_duration.append((path['timestamp'][-1] - path['timestamp'][0]).total_seconds() / (3600 * 24))
        plt.hist(path_duration, nr_bars)
        plt.title('Conversion path duration')
        plt.xlabel('Length [days]')
        plt.ylabel('Frequency')
        plt.show()

    def plot_channel_conversion_frequency_GA(self, normalize=True):
        non_conversion_paths = self.get_non_conversion_paths()
        conversion_paths_not_last = self.get_conversion_paths_not_last()
        conversion_paths_last = self.get_conversion_paths_last()

        occur_per_channel_non_conv = non_conversion_paths['source_medium'].value_counts()
        occur_per_channel_conv_last = conversion_paths_last['source_medium'].value_counts()
        occur_per_channel_conv_not_last = conversion_paths_not_last['source_medium'].value_counts()

        if normalize:
            occur_per_channel_conv_last = occur_per_channel_conv_last / occur_per_channel_conv_last.sum()
            occur_per_channel_non_conv = occur_per_channel_non_conv / occur_per_channel_non_conv.sum()
            occur_per_channel_conv_not_last = occur_per_channel_conv_not_last / occur_per_channel_conv_not_last.sum()

        df = pd.DataFrame({"Conversions last": occur_per_channel_conv_last,
                           "Conversions not last": occur_per_channel_conv_not_last,
                           "Non-conversion any time": occur_per_channel_non_conv})
        ax = df.plot.bar(title="Source/medium occurences in paths")
        ax.set_xlabel("Source / Medium")
        if normalize:
            ax.set_ylabel("Proportion")
        else:
            ax.set_ylabel("Counts")
        plt.tight_layout()
        plt.show()

    def plot_perc_occur_conv_spend(self):
        get_conversion_paths = self.get_conversion_paths()
        cost_per_source_medium = get_conversion_paths.groupby(['source_medium']).agg('sum')['cost']
        cost_per_source_medium_perc = 100 * (cost_per_source_medium / cost_per_source_medium.sum())
        conv_per_source_medium_perc = 100 * get_conversion_paths['source_medium'].value_counts() / len(get_conversion_paths['source_medium'])

        conv_cost_df_perc = pd.concat([conv_per_source_medium_perc, cost_per_source_medium_perc], axis=1)
        conv_cost_df_perc.rename(columns={"source_medium": "Conversion path %", "cost": "Spend %"}, inplace=True)

        conv_cost_df_perc.plot(y=["Conversion path %", "Spend %"], kind="bar")
        plt.tight_layout()
        plt.ylabel('%')
        plt.title('Spend percentage vs percentage of occurring in conversion paths')
        plt.show()

    def plot_age_dist_MP(self):
        fig, ax = plt.subplots()
        self.MP_df['age'].value_counts(sort=False).plot(ax=ax, kind='bar')
        plt.title('Age distribution - Hedvig users')
        plt.show()

    def plot_premium_dist_MP(self):
        fig, ax = plt.subplots()
        self.MP_df = self.MP_df.sort_values(by=['premium'], ascending=False)
        self.MP_df['premium'].value_counts(sort=False).plot(ax=ax, kind='bar')
        plt.title('Premium distribution - Hedvig users')
        plt.show()

    def plot_premium_age_MP(self):
        plt.scatter(self.MP_df['age'], self.MP_df['premium'])
        plt.title("Premium over age")
        plt.xlabel("age")
        plt.ylabel("premium")
        plt.show()

    def plot_user_conversions_not_last_against_source_hist(self, user_feature, nr_channels=5):
        conversion_paths_not_last_df = self.get_conversion_paths_not_last()
        channels = conversion_paths_not_last_df['source_medium'].value_counts()[:nr_channels]
        user_data_per_channel = []
        for channel, _ in channels.iteritems():
            client_indexes = conversion_paths_not_last_df.loc[
                conversion_paths_not_last_df['source_medium'] == channel].index
            client_ids = [client_id[0] for client_id in client_indexes]
            user_data = self.converted_clients_df[self.converted_clients_df['client_id'].isin(client_ids)][user_feature]
            user_data_per_channel.append(user_data)

        labels = list(channels.index)
        plt.hist(user_data_per_channel, bins=5, density=False, label=labels)
        plt.legend()
        plt.title(user_feature.capitalize() + ' per non-last conversion channel')
        plt.xlabel(user_feature.capitalize())
        plt.ylabel('Counts')
        plt.show()

    def plot_user_conversions_not_last_against_source_curve(self, column, nr_channels=3, bandwidth=10,
                                                            transparency=0.4):
        conversion_paths_not_last_df = self.get_conversion_paths_not_last()
        channels = conversion_paths_not_last_df['source_medium'].value_counts()[:nr_channels]
        channel_idx = 0
        for channel, _ in channels.iteritems():
            client_indexes = conversion_paths_not_last_df.loc[
                conversion_paths_not_last_df['source_medium'] == channel].index
            client_ids = [client_id[0] for client_id in client_indexes]
            user_data = self.converted_clients_df[self.converted_clients_df['client_id'].isin(client_ids)][column]
            x_plot = np.linspace(min(user_data) - 3 * bandwidth, max(user_data) + 3 * bandwidth, 1000)[:, np.newaxis]
            plt.fill(x_plot[:, 0], np.exp(
                KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(
                    np.array(user_data).reshape(-1, 1)).score_samples(x_plot)),
                     alpha=transparency, label=channel, fc=plt.get_cmap('tab10')(channel_idx))
            channel_idx += 1

        plt.title('')
        plt.legend()
        plt.title(column.capitalize() + ' per non-last conversion channel')
        plt.xlabel(column.capitalize())
        plt.ylabel('Proportion')
        plt.show()

    def plot_cpc_per_channel_over_time(self):
        funnel_temp_df = self.funnel_df
        funnel_temp_df.set_index(['Date', 'Traffic_source'])
        fig, ax = plt.subplots(figsize=(15, 7))
        funnel_temp_df.groupby(['Date', 'Traffic_source']).sum()['cpc'].unstack().plot(ax=ax, rot=90)
        plt.title('Mean cpc per day per channel')
        plt.ylabel('CPC [SEK]')
        plt.grid()
        plt.show()

    def show_interesting_results_MP(self):
        self.plot_premium_age_MP()
        self.plot_age_dist_MP()
        self.plot_premium_dist_MP()

    def show_interesting_results_GA(self):
        self.plot_perc_occur_conv_spend()
        self.plot_channel_conversion_frequency_GA()
        self.plot_path_length_GA()
        self.plot_path_duration_GA()

    def show_interesting_results_combined(self):
        self.plot_user_conversions_not_last_against_source_curve('age', nr_channels=3, bandwidth=0.5)

    def show_interesting_results_funnel(self):
        self.plot_cpc_per_channel_over_time()


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.options.display.width = 0

    file_path_mp = '../Data/Mixpanel_data_2021-02-11.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=1, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=10, hour=23, minute=59, tz='UTC')

    descriptives = Descriptives(start_date, end_date, file_path_mp)
    descriptives.show_interesting_results_GA()