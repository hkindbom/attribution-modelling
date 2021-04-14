import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KernelDensity
from DataProcessing import DataProcessing
matplotlib.rcParams['font.serif'] = "Times New Roman"
matplotlib.rcParams['font.family'] = "serif"
matplotlib.rcParams.update({'font.size': 15})


class Descriptives:
    def __init__(self, start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mp=None,
                 nr_top_ch=10000):
        self.start_date_data = start_date_data
        self.end_date_data = end_date_data
        self.start_date_cohort = start_date_cohort
        self.end_date_cohort = end_date_cohort
        self.data_processing = DataProcessing(self.start_date_data, self.end_date_data, start_date_cohort,
                                              end_date_cohort, file_path_mp, nr_top_ch=nr_top_ch)
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

    def corr_metric(self, x, y):
        cov = np.cov(x,y)[0][1]
        return cov / (np.std(x) * np.std(y)) if cov !=0 else 0

    def show_ctrl_vars_corr(self, ctrl_vars_list, threshold_corr=0., threshold_prop=0.):
        correlation_df = pd.DataFrame()
        channels = self.GA_df['source_medium'].value_counts().index
        for channel in channels:
            ch_sessions_df = self.GA_df[self.GA_df['source_medium'] == channel]
            channel_conv_vector = ch_sessions_df['converted_eventually'].to_numpy()
            for ctrl_var in ctrl_vars_list:
                ctrl_var_values = self.GA_df[ctrl_var].value_counts().index
                for ctrl_var_value in ctrl_var_values:
                    ctrl_vector = np.array(ch_sessions_df[ctrl_var] == ctrl_var_value).astype(int)
                    corr_coef = self.corr_metric(channel_conv_vector, ctrl_vector)
                    prop_ctrl_var_in_data = len(self.GA_df[self.GA_df[ctrl_var] == ctrl_var_value])/len(self.GA_df) # Length is on session level
                    prop_ctrl_var_in_ch = np.sum(ctrl_vector)/len(ctrl_vector)
                    if prop_ctrl_var_in_data > threshold_prop and abs(corr_coef) > threshold_corr:
                        result_dict = {'channel': channel, 'ctrl-var': ctrl_var, 'ctrl-var-value': ctrl_var_value,
                                       'corr coef': corr_coef, 'prop ctrl var in data': prop_ctrl_var_in_data,
                                       'prop ctrl variable in channel': prop_ctrl_var_in_ch}
                        correlation_df = correlation_df.append(result_dict, ignore_index=True)
        correlation_df = correlation_df.reindex(correlation_df['corr coef'].abs().sort_values(ascending=False).index)
        print(correlation_df)

    def plot_path_length_GA(self):
        self.count_nr_ch_in_path()
        conversion_paths = self.get_conversion_paths()
        path_lengths = []
        for client, path_df in conversion_paths.groupby(level=0):
            path_lengths.append(len(path_df))
        temp_df = pd.DataFrame({'freq': path_lengths})
        temp_df.groupby('freq', as_index=False).size().plot(x='freq', y='size', kind='bar', legend=False)
        plt.title('Conversion path lengths')
        plt.xlabel('Length [clicks]')
        plt.ylabel('Positive Customer Journeys')
        plt.show()

    def count_nr_ch_in_path(self):
        conversion_paths = self.get_conversion_paths()
        count_diff_ch = 0
        count_tot_long_ch = 0
        for client, path_df in conversion_paths.groupby(level=0):
            if len(path_df)>1:
                count_tot_long_ch += 1
                if len(path_df['source_medium'].unique())>1:
                    count_diff_ch += 1
        prop = 100 * count_diff_ch / len(self.get_conversion_paths_last())
        print('Nr of paths of len>1 with different channels', count_diff_ch, '(', prop, '% of all positive conversions)')

    def plot_path_duration_GA(self, nr_bars=20):
        csfont = {'fontname': 'Times New Roman'}
        conversion_paths = self.get_conversion_paths()
        path_duration = []
        for client, path in conversion_paths.groupby(level=0):
            path_duration.append((path['timestamp'][-1] - path['timestamp'][0]).total_seconds() / (3600 * 24))
        plt.hist(path_duration, nr_bars)
        plt.title('Conversion path duration',**csfont)
        plt.xlabel('Length [days]',**csfont)
        plt.ylabel('Positive Customer Journeys',**csfont)
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
        cost_per_source_medium = self.GA_df.groupby(['source_medium']).agg('sum')['cost']
        # print(cost_per_source_medium)
        cost_per_source_medium_perc = 100 * (cost_per_source_medium / cost_per_source_medium.sum())
        conversion_paths = self.get_conversion_paths()

        # Note! % conversion for each channel is between 0 and 100 and % spend is between 0 and 100 for all channels together (only based on cpc)
        conv_paths_count = conversion_paths.reset_index().groupby('client_id')['source_medium'].value_counts().to_frame('path_count').reset_index()
        conv_per_source_medium_perc = 100 * conv_paths_count['source_medium'].value_counts() / len(self.get_conversion_paths_last()['source_medium'])

        conv_cost_df_perc = pd.concat([conv_per_source_medium_perc, cost_per_source_medium_perc], axis=1).fillna(0)
        conv_cost_df_perc.rename(columns={"source_medium": "% of Conversion Paths", "cost": "% of Tot Spend"}, inplace=True)

        conv_cost_df_perc.plot(y=["% of Conversion Paths", "% of Tot Spend"], kind="bar")
        plt.tight_layout()
        plt.ylabel('%')
        plt.title(f"Spend % vs occurrences % in conversion paths during "
                  f"{self.start_date_cohort.year}-{self.start_date_cohort.month}-{self.start_date_cohort.day} to "
                  f"{self.end_date_data.year}-{self.end_date_data.month}-{self.end_date_data.day}")
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

    def plot_user_conversions_not_last_against_source_curve(self, column, nr_channels=3, bandwidth=10.,
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

    file_path_mp = '../Data/Mixpanel_data_2021-03-19.csv'
    start_date_data = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    start_date_cohort = pd.Timestamp(year=2021, month=2, day=24, hour=0, minute=0, tz='UTC')
    end_date_cohort = pd.Timestamp(year=2021, month=3, day=17, hour=23, minute=59, tz='UTC')
    end_date_data = pd.Timestamp(year=2021, month=4, day=7, hour=23, minute=59, tz='UTC')

    nr_top_ch = 10

    descriptives = Descriptives(start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mp, nr_top_ch)
    ctrl_vars_list = ['device_category', 'city', 'browser', 'operating_system']
    descriptives.show_ctrl_vars_corr(ctrl_vars_list, threshold_prop=0.1)