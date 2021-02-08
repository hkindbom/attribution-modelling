import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from datetime import timedelta

from googleapiclient.discovery import build
#import googleapiclient
from oauth2client.service_account import ServiceAccountCredentials

class ApiDataGA:
    def __init__(self):
        self.SCOPES = ['https://www.googleapis.com/auth/analytics.readonly']
        self.KEY_FILE_LOCATION = '../API/Master-Thesis-GA-api-b6dc4fc6d4dd.json'
        self.VIEW_ID = '229972923'

    def initialize_analyticsreporting(self):
        """Initializes an Analytics Reporting API V4 service object.
        Returns: An authorized Analytics Reporting API V4 service object.
        """
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.KEY_FILE_LOCATION, self.SCOPES)

        # Build the service object.
        analytics = build('analyticsreporting', 'v4', credentials=credentials)
        return analytics

    def get_report(self, analytics):
        """Queries the Analytics Reporting API V4.
        Args:
          analytics: An authorized Analytics Reporting API V4 service object.
        Returns:
          The Analytics Reporting API V4 response.
        """
        return analytics.reports().batchGet(
            body={
                'reportRequests': [
                    {
                        'viewId': self.VIEW_ID,
                        'dateRanges': [{'startDate': '7daysAgo', 'endDate': 'today'}],
                        'metrics': [{'expression': 'ga:sessions'}],
                        'dimensions': [{'name': 'ga:country'}]
                    }]
            }
        ).execute()

    def print_response(self,response):
        """Parses and prints the Analytics Reporting API V4 response.
        Args:
          response: An Analytics Reporting API V4 response.
        """
        for report in response.get('reports', []):
            columnHeader = report.get('columnHeader', {})
            dimensionHeaders = columnHeader.get('dimensions', [])
            metricHeaders = columnHeader.get('metricHeader', {}).get('metricHeaderEntries', [])

            for row in report.get('data', {}).get('rows', []):
                dimensions = row.get('dimensions', [])
                dateRangeValues = row.get('metrics', [])

                for header, dimension in zip(dimensionHeaders, dimensions):
                    print(header + ': ', dimension)

                for i, values in enumerate(dateRangeValues):
                    print('Date range:', str(i))
                    for metricHeader, value in zip(metricHeaders, values.get('values')):
                        print(metricHeader.get('name') + ':', value)

    def main(self):
        analytics = self.initialize_analyticsreporting()
        response = self.get_report(analytics)
        self.print_response(response)

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

    def process_mixpanel_data(self, start_time = pd.Timestamp(year = 2021, month = 2, day = 1, tz = 'UTC'),
                              convert_to_float = True, market = 'SE'):
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
                                '$properties.$termination_date': 'termination_date',
                                '$properties.$timezone': 'timezone',
                                '$properties.$zip_code': 'zip'})
        ## Be aware! Check time zones Daylight Savings Time (GA vs. MP)
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce').dt.tz_localize('Europe/Oslo', ambiguous = False)

        df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce').dt.tz_localize('Europe/Oslo', ambiguous = False)

        df = df.loc[df['signup_time'] >= start_time]  ## Filter by signup time

        df = df.loc[df['market'] == market]  ## Filter by regional market

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

    def estimate_client_LTV(self, w_premium = 1, w_nr_co_insured = - 5):
        self.converted_clients_df['LTV'] = 0
        for index, client in self.converted_clients_df.iterrows():
            self.converted_clients_df.loc[index, 'LTV'] = client['premium'] * w_premium \
                                                          + client['nr_co_insured'] * w_nr_co_insured
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

    def process_all(self, start_time):
        self.process_individual_data()
        self.group_by_client_id()
        self.remove_post_conversion()
        self.process_mixpanel_data(start_time)
        self.create_converted_clients_df()
        self.estimate_client_LTV()

class Descriptives:
    def __init__(self, start_time, file_path_GA_main = None, file_path_GA_secondary = None, file_path_mp = None):
        self.start_time = start_time
        self.data_processing = DataProcessing(file_path_GA_main, file_path_GA_secondary, file_path_mp)
        self.GA_df = None
        self.MP_df = None
        self.converted_clients_df = None
        self.read_data()

    def read_data(self):
        self.data_processing.process_all(start_time)
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

    def plot_user_conversions_not_last_against_source_hist(self, column, nr_channels = 5):
        conversion_paths_not_last_df = self.GA_df.loc[(self.GA_df['converted_eventually'] == 1) &
                                                   (self.GA_df['conversion'] == 0)]
        channels = conversion_paths_not_last_df['source_medium'].value_counts()[:nr_channels]
        #channels = channels.append(pd.Series({'studentkortet / partnership': 0}))
        plot_data_per_channel = []
        for channel, _ in channels.iteritems():
            client_indexes = conversion_paths_not_last_df.loc[conversion_paths_not_last_df['source_medium'] == channel].index
            client_ids = [client_id[0] for client_id in client_indexes]
            user_data = self.converted_clients_df[self.converted_clients_df['client_id'].isin(client_ids)][column]
            plot_data_per_channel.append(user_data)

        labels = list(channels.index)
        plt.hist(plot_data_per_channel, bins=5, density=False, label=labels)
        plt.legend()
        plt.title(column.capitalize() + ' per non-last conversion channel')
        plt.xlabel(column.capitalize())
        plt.ylabel('Counts')
        plt.show()

    def plot_user_conversions_not_last_against_source_curve(self, column, nr_channels = 3, bandwidth = 10, transparency = 0.4):
        conversion_paths_not_last_df = self.GA_df.loc[(self.GA_df['converted_eventually'] == 1) &
                                                      (self.GA_df['conversion'] == 0)]
        channels = conversion_paths_not_last_df['source_medium'].value_counts()[:nr_channels]
        channel_idx = 0
        #channels = channels[::-1]
        for channel, _ in channels.iteritems():
            client_indexes = conversion_paths_not_last_df.loc[
                conversion_paths_not_last_df['source_medium'] == channel].index
            client_ids = [client_id[0] for client_id in client_indexes]
            user_data = self.converted_clients_df[self.converted_clients_df['client_id'].isin(client_ids)][column]
            x_plot = np.linspace(min(user_data)-3*bandwidth, max(user_data)+3*bandwidth, 1000)[:, np.newaxis]
            plt.fill(x_plot[:, 0], np.exp(
                KernelDensity(kernel = 'gaussian', bandwidth = bandwidth).fit(
                    np.array(user_data).reshape(-1, 1)).score_samples(x_plot)),
                     alpha = transparency, label = channel, fc = plt.get_cmap('tab10')(channel_idx))
            channel_idx += 1

        plt.title('')
        plt.legend()
        plt.title(column.capitalize() + ' per non-last conversion channel')
        plt.xlabel(column.capitalize())
        plt.ylabel('Proportion')
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
        #self.plot_user_conversions_not_last_against_source('age')
        self.plot_user_conversions_not_last_against_source_curve('premium', 3, 10)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_GA_main = '../Data/Analytics_sample_1.csv'
    file_path_GA_secondary = '../Data/Analytics_sample_2.csv'
    file_path_mp = '../Data/Mixpanel_data_2021-02-05.csv'
    start_time = pd.Timestamp(year=2021, month=2, day=1, tz='UTC')

    api = ApiDataGA()
    descriptives = Descriptives(start_time, file_path_GA_main, file_path_GA_secondary, file_path_mp)
    descriptives = descriptives.show_interesting_results_combined()