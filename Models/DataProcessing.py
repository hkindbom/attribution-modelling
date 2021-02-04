import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

class DataProcessing:
    def __init__(self, file_path_GA_main = None, file_path_GA_secondary = None, file_path_mixpanel = None,
                 file_path_GA_aggregated = None, save_to_path = None):
        self.GA_df = None
        self.MP_df = None
        self.converted_clients_df = None
        self.GA_aggregated_df = None
        self.file_path_GA_main = file_path_GA_main
        self.file_path_GA_secondary = file_path_GA_secondary
        self.file_path_mixpanel = file_path_mixpanel
        self.file_path_GA_aggregated = file_path_GA_aggregated
        self.save_to_path = save_to_path

    def process_individual_data(self):
        df1 = pd.read_csv(self.file_path_GA_main, header=5, dtype={'cllientId': str}) ##Remember to change header=5
        df2 = pd.read_csv(self.file_path_GA_secondary, header=5, dtype={'cllientId': str})
        cols_to_use = df2.columns.difference(df1.columns)
        df = pd.merge(df1, df2[cols_to_use], left_index=True, right_index=True, how='outer')

        df = df.rename(columns={'cllientId': 'client_id',
                                'sessionId': 'session_id',
                                'Campaign': 'campaign',
                                'Sessions': 'sessions',
                                'hit timestamp': 'timestamp',
                                'Source / Medium': 'source_medium',
                                'Signed Customer (Goal 1 Completions)': 'conversion',
                                'Signed Customer (Goal 1 Value)': 'conversion_value',
                                'Device Category': 'device_category'})
        ## Be aware! Check time zones Daylight Savings Time (GA vs. MP)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['conversion_value'] = df['conversion_value'].str.replace('SEK','').astype(float)

        self.GA_df = df

    def group_by_client_id(self):
        df = self.GA_df.sort_values(by=['client_id', 'timestamp'], ascending=True)
        df = df.set_index(['client_id', 'session_id'])
        df['converted_eventually'] = 0

        for client, temp_df in df.groupby(level=0):
            if max(temp_df['conversion']) == 1:
                df.loc[str(client), 'converted_eventually'] = 1

        self.GA_df = df

    def remove_post_conversion(self):
        df = self.GA_df
        conversion_sessions = df.loc[df['converted_eventually'] == 1]
        sessions_to_delete = df[0:0]

        for client, cust_journey in conversion_sessions.groupby(level=0):
            if len(cust_journey) > 1:
                conversion_time = cust_journey[cust_journey['conversion'] == 1].iloc[-1]['timestamp']
                post_conversion = cust_journey.loc[cust_journey['timestamp'] > conversion_time]
                if not post_conversion.empty:
                    sessions_to_delete = sessions_to_delete.append(post_conversion)
        self.GA_df = pd.concat([df, sessions_to_delete, sessions_to_delete]).drop_duplicates(keep=False)

    def process_aggregated_data(self):
        df = pd.read_csv(self.file_path_GA_aggregated, header=5)
        df = df.rename(columns={'Väg till konvertering per källa': 'path',
                                'Konverteringar': 'total_conversions',
                                'Konverteringsvärde': 'total_conversion_value'})
        df['total_null'] = 0
        df['total_conversions'] = df['total_conversions'].str.replace('\s+', '', regex=True).astype(int)
        df['total_conversion_value'] = df['total_conversion_value'].str.rstrip('kr').\
            str.replace(',','.', regex=True).str.replace('\s+','', regex=True).astype(float)
        self.GA_aggregated_df = df

    def process_mixpanel_data(self, start_time = pd.Timestamp(year = 2021, month = 2, day = 1, tz='UTC'),
                              convert_to_float=True):
        df = pd.read_csv(self.file_path_mixpanel)

        df = df[df['$properties.$created_at'] != 'undefined'] # Filter out non-singups

        df = df.rename(columns={'$distinct_id': 'user_id',
                                '$properties.$city': 'city',
                                '$properties.$initial_referring_domain': 'source',
                                '$properties.$created_at': 'signup_time',
                                '$properties.$premium': 'premium',
                                '$properties.$age': 'age',
                                '$properties.$line_of_business': 'business',
                                '$properties.$market': 'market',
                                '$properties.$number_co_insured': 'nr_co_insured',
                                '$properties.$number_failed_charges': 'nr_failed_charges',
                                '$properties.$street': 'street',
                                '$properties.$termination_date': 'termination_date',
                                '$properties.$timezone': 'timezone',
                                '$properties.$zip_code': 'zip'})
        ## Be aware! Check time zones Daylight Savings Time (GA vs. MP)
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce').dt.tz_localize('Europe/Oslo', ambiguous = False)

        df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce').dt.tz_localize('Europe/Oslo', ambiguous = False)

        df = df.loc[df['signup_time'] >= start_time]

        if convert_to_float:
            df['premium'] = pd.to_numeric(df['premium'], errors='coerce')
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['zip'] = pd.to_numeric(df['zip'], errors='coerce')
            df['nr_failed_charges'] = pd.to_numeric(df['nr_failed_charges'], errors='coerce').fillna(0)
            df['nr_co_insured'] = pd.to_numeric(df['nr_co_insured'], errors='coerce').fillna(0)

        self.MP_df = df

    def create_converted_clients_df(self, minute_margin = 1.5, premium_margin = 10):
        self.converted_clients_df = pd.DataFrame(columns=['client_id'] + list(self.MP_df.columns))

        conversion_sessions_df = self.GA_df.loc[self.GA_df['conversion'] == 1]
        for client, conversion_session in conversion_sessions_df.iterrows():
            self.match_client(client, conversion_session, minute_margin, premium_margin)
        nr_conversions = len(self.GA_df.loc[self.GA_df['conversion'] == 1])
        print('Matched ', len(self.converted_clients_df), ' users of ', nr_conversions)


    def match_client(self, client, conversion_session, minute_margin, premium_margin):
        conversion_time = pd.to_datetime(conversion_session['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), utc=True)
        starttime = conversion_time - timedelta(minutes = minute_margin)
        endtime = conversion_time + timedelta(minutes = minute_margin)

        mixpanel_user = self.MP_df.loc[(self.MP_df['signup_time'] >= starttime) &
                                       (self.MP_df['signup_time'] <= endtime)]

        if len(mixpanel_user) == 0:
            return

        if len(mixpanel_user) == 1:
            self.append_matching_client(mixpanel_user, client)
            return

        if len(mixpanel_user) > 1:  ## If multiple measures
            lower_premium = conversion_session['conversion_value'] - premium_margin
            upper_premium = conversion_session['conversion_value'] + premium_margin
            mixpanel_user = mixpanel_user.loc[(mixpanel_user['premium'] >= lower_premium) &
                                              (mixpanel_user['premium'] <= upper_premium)]
            if len(mixpanel_user) == 1:
                self.append_matching_client(mixpanel_user, client)
                return

    def append_matching_client(self, mixpanel_user, client):
        mixpanel_user.insert(0, 'client_id', client[0])
        self.converted_clients_df = self.converted_clients_df.append(mixpanel_user, ignore_index=True)


    def save_to_csv(self):
        self.GA_df.to_csv(self.save_to_path, sep=',')

    def get_GA_df(self):
        return self.GA_df

    def get_MP_df(self):
        return self.MP_df

    def get_GA_aggr_df(self):
        return self.GA_aggregated_df

    def get_converted_clients_df(self):
        return self.converted_clients_df

    def get_client_mixpanel_info(self, client):
        return self.converted_clients_df.loc[self.converted_clients_df['client_id'] == client]

class Descriptives:

    def __init__(self, start_time, file_path_GA_main = None, file_path_GA_secondary = None, file_path_mp = None):
        self.start_time = start_time
        self.data_processing = DataProcessing(file_path_GA_main, file_path_GA_secondary, file_path_mp)
        self.GA_df = None
        self.MP_df = None
        self.converted_clients_df = None
        self.read_data()

    def read_data(self):
        self.data_processing.process_individual_data()
        self.data_processing.group_by_client_id()
        self.data_processing.remove_post_conversion()
        self.data_processing.process_mixpanel_data(self.start_time)
        self.data_processing.create_converted_clients_df()

        self.GA_df = self.data_processing.get_GA_df()
        self.MP_df = self.data_processing.get_MP_df()
        self.converted_clients_df = self.data_processing.get_converted_clients_df()

    def plot_path_length_GA(self):
        conversion_paths = self.GA_df.loc[self.GA_df['converted_eventually'] == 1]
        path_lengths = []
        for client, path in conversion_paths.groupby(level=0):
            path_lengths.append(len(path))
        temp_df = pd.DataFrame({'freq': path_lengths})
        temp_df.groupby('freq', as_index=False).size().plot(x='freq', y='size', kind='bar')
        plt.title('Conversion path lengths')
        plt.xlabel('Length')
        plt.ylabel('Frequency')
        plt.show()

    def plot_path_duration_GA(self, nr_bars = 10):
        conversion_paths = self.GA_df.loc[self.GA_df['converted_eventually'] == 1]
        path_duration = []
        for client, path in conversion_paths.groupby(level=0):
            path_duration.append((path['timestamp'][-1] - path['timestamp'][0]).total_seconds()/(3600*24))
        plt.hist(path_duration, nr_bars)
        plt.title('Conversion path duration')
        plt.xlabel('Length [days]')
        plt.ylabel('Frequency')
        plt.show()

    def plot_channel_conversion_frequency_GA(self, normalize = True):
        non_conversion_paths = self.GA_df.loc[self.GA_df['converted_eventually'] == 0]
        conversion_paths = self.GA_df.loc[self.GA_df['converted_eventually'] == 1]
        conversion_paths_not_last = conversion_paths.loc[conversion_paths['conversion'] == 0]
        conversion_paths_last = conversion_paths.loc[conversion_paths['conversion'] == 1]

        occur_per_channel_non_conv = non_conversion_paths['source_medium'].value_counts()
        occur_per_channel_conv_last = conversion_paths_last['source_medium'].value_counts()
        occur_per_channel_conv_not_last = conversion_paths_not_last['source_medium'].value_counts()

        if normalize:
            occur_per_channel_conv_last = occur_per_channel_conv_last / occur_per_channel_conv_last.sum()
            occur_per_channel_non_conv = occur_per_channel_non_conv/occur_per_channel_non_conv.sum()
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

    def plot_age_dist_MP(self):
        fig, ax = plt.subplots()
        self.MP_df['age'].value_counts().plot(ax=ax, kind='bar')
        plt.title('Age distribution - Hedvig users')
        plt.show()

    def plot_premium_dist_MP(self):
        fig, ax = plt.subplots()
        self.MP_df = self.MP_df.sort_values(by=['premium'], ascending=False)
        self.MP_df['premium'].value_counts().plot(ax=ax, kind='bar')
        plt.title('Premium distribution - Hedvig users')
        plt.show()

    def plot_premium_age_MP(self):
        plt.scatter(self.MP_df['age'], self.MP_df['premium'])
        plt.title("Premium over age")
        plt.xlabel("age")
        plt.ylabel("premium")
        plt.show()

    def show_interesting_results_MP(self):
        self.plot_premium_age_MP()
        self.plot_age_dist_MP()
        self.plot_premium_dist_MP()

    def show_interesting_results_GA(self):
        self.plot_channel_conversion_frequency_GA()
        self.plot_path_length_GA()
        self.plot_path_duration_GA()

    def show_interesting_results_combined(self):
        pass

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_GA_main = '../Data/Analytics_sample_1.csv'
    file_path_GA_secondary = '../Data/Analytics_sample_2.csv'
    file_path_mp = '../Data/Mixpanel_data_2021-02-04.csv'

    #data_processing.save_to_csv()
    #df = data_processing.get_mixpanel_df()

    descriptives = Descriptives(pd.Timestamp(year = 2021, month = 2, day = 1, tz='UTC'), file_path_GA_main, file_path_GA_secondary, file_path_mp)
    descriptives.show_interesting_results_GA()
