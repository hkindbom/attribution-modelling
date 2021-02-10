import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from datetime import timedelta
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account
from google.cloud import bigquery, bigquery_storage

class ApiDataBigQuery:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.funnel_df = None
        self.fetch_BQ()

    # Be Aware! Can only handle max one month at a time
    def fetch_BQ(self):
        credentials = service_account.Credentials.from_service_account_file('../API/BQ_api.json')
        bqclient = bigquery.Client(credentials=credentials, project=credentials.project_id)
        bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

        query_string = f"SELECT * FROM funnel-integration.Marketing_Spend.marketing_spend_monthly_" \
                       f"{self.start_date.year}_{str(self.start_date.month).zfill(2)} " \
                       f"WHERE Date <= DATE ({self.end_date.year}, {self.end_date.month}, {self.end_date.day}) " \
                       f"AND Date >= DATE ({self.start_date.year}, {self.start_date.month}, {self.start_date.day})"

        self.funnel_df = (
            bqclient.query(query_string)
                .result()
                .to_dataframe(bqstorage_client=bqstorageclient)
        )

    def get_funnel_df(self):
        return self.funnel_df


class ApiDataGA:
    def __init__(self, start_date, end_date):
        self.scopes = ['https://www.googleapis.com/auth/analytics.readonly']
        self.key_file_location = '../API/ga-api.json'
        self.view_id = open('../API/view_id.txt', 'r').readline()
        self.analytics = None
        self.GA_api_df = None
        self.start_date = start_date
        self.end_date = end_date

    def initialize_analyticsreporting(self):  # Initializes an Analytics Reporting API V4 service object.
        credentials = ServiceAccountCredentials.from_json_keyfile_name(self.key_file_location, self.scopes)
        analytics = build('analyticsreporting', 'v4', credentials=credentials)
        self.analytics = analytics

    def create_report_df(self):  # Queries the Analytics Reporting API V4. Returns the API response.
        metrics = ['ga:sessions', 'ga:goal1Completions', 'ga:goal1Value']
        # dimension6 = cllientId, dimension7 = sessionId, dimension8 = hit timestamp
        dims = ['ga:dimension6', 'ga:dimension7', 'ga:dimension8', 'ga:campaign',
                'ga:sourcemedium', 'ga:source', 'ga:devicecategory']
        data = self.analytics.reports().batchGet(
            body={
                'reportRequests': [
                    {
                        'viewId': self.view_id,
                        'dateRanges': [
                            {'startDate': str(self.start_date.date()), 'endDate': str(self.end_date.date())}],
                        'metrics': [{'expression': exp} for exp in metrics],
                        'dimensions': [{'name': name} for name in dims],
                        'pageSize': 100000,  # max nr of query results allowed by api
                        'includeEmptyRows': False,
                    }]
            }
        ).execute()

        data_dic = {f"{i}": [] for i in dims + metrics}
        for report in data.get('reports', []):
            rows = report.get('data', {}).get('rows', [])
            for row in rows:
                for i, key in enumerate(dims):
                    data_dic[key].append(row.get('dimensions', [])[i])  # Get dimensions
                dateRangeValues = row.get('metrics', [])
                for values in dateRangeValues:
                    all_values = values.get('values', [])  # Get metric values
                    for i, key in enumerate(metrics):
                        data_dic[key].append(all_values[i])

        GA_api_df = pd.DataFrame(data=data_dic)
        GA_api_df.columns = [col.split(':')[-1] for col in GA_api_df.columns]

        self.GA_api_df = GA_api_df

    def get_GA_df(self):
        return self.GA_api_df

    def initialize_api(self):
        self.initialize_analyticsreporting()
        self.create_report_df()


class DataProcessing:
    def __init__(self, start_date, end_date, file_path_mixpanel=None, file_path_GA_aggregated=None, save_to_path=None):
        self.start_date = start_date
        self.end_date = end_date
        self.GA_df = None
        self.MP_df = None
        self.converted_clients_df = None
        self.GA_aggregated_df = None
        self.file_path_mixpanel = file_path_mixpanel
        self.file_path_GA_aggregated = file_path_GA_aggregated
        self.save_to_path = save_to_path

    def process_individual_data(self):
        GA_api = ApiDataGA(self.start_date, self.end_date)
        GA_api.initialize_api()
        GA_api_df = GA_api.get_GA_df()
        GA_api_df = GA_api_df.rename(columns={'dimension6': 'client_id',
                                              'dimension7': 'session_id',
                                              'campaign': 'campaign',
                                              'sessions': 'sessions',
                                              'dimension8': 'timestamp',
                                              'sourcemedium': 'source_medium',
                                              'goal1Completions': 'conversion',
                                              'goal1Value': 'conversion_value',
                                              'devicecategory': 'device_category'})

        # Be aware! Check time zones Daylight Savings Time (GA vs. MP)
        GA_api_df['timestamp'] = pd.to_datetime(GA_api_df['timestamp'], utc=True)
        GA_api_df['conversion'] = GA_api_df['conversion'].astype(int)
        GA_api_df['conversion_value'] = GA_api_df['conversion_value'].astype(float)
        GA_api_df['sessions'] = GA_api_df['sessions'].astype(int)

        self.GA_df = GA_api_df

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
        df['total_conversion_value'] = df['total_conversion_value'].str.rstrip('kr'). \
            str.replace(',', '.', regex=True).str.replace('\s+', '', regex=True).astype(float)
        self.GA_aggregated_df = df

    def process_mixpanel_data(self, convert_to_float=True, market='SE'):
        df = pd.read_csv(self.file_path_mixpanel)

        df = df[df['$properties.$created_at'] != 'undefined']  # Filter out non-singups

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
        # Be aware! Check time zones Daylight Savings Time (GA vs. MP)
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce').dt.tz_localize('Europe/Oslo',
                                                                                              ambiguous=False)

        df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce').dt.tz_localize('Europe/Oslo',
                                                                                                        ambiguous=False)

        df = df.loc[(df['signup_time'] >= self.start_date) &
                    (df['signup_time'] <= self.end_date)]  # Filter by signup time
        df = df.loc[df['market'] == market]  # Filter by regional market

        if convert_to_float:
            df['premium'] = pd.to_numeric(df['premium'], errors='coerce')
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['zip'] = pd.to_numeric(df['zip'], errors='coerce')
            df['nr_failed_charges'] = pd.to_numeric(df['nr_failed_charges'], errors='coerce').fillna(0)
            df['nr_co_insured'] = pd.to_numeric(df['nr_co_insured'], errors='coerce').fillna(0)

        self.MP_df = df

    def create_converted_clients_df(self, minute_margin=1.5, premium_margin=10):
        self.converted_clients_df = pd.DataFrame(columns=['client_id'] + list(self.MP_df.columns))

        conversion_sessions_df = self.GA_df.loc[self.GA_df['conversion'] == 1]
        for client, conversion_session in conversion_sessions_df.iterrows():
            self.match_client(client, conversion_session, minute_margin, premium_margin)
        nr_conversions = len(self.GA_df.loc[self.GA_df['conversion'] == 1])
        print('Matched ' + str(len(self.converted_clients_df)) + ' users of ' + str(nr_conversions) +
              ' (' + str(round(len(self.converted_clients_df) / nr_conversions * 100, 2)) + ' %)')

    def match_client(self, client, conversion_session, minute_margin, premium_margin):
        conversion_time = pd.to_datetime(conversion_session['timestamp'].strftime('%Y-%m-%d %H:%M:%S'), utc=True)
        start_time = conversion_time - timedelta(minutes=minute_margin)
        end_time = conversion_time + timedelta(minutes=minute_margin)

        mixpanel_user = self.MP_df.loc[(self.MP_df['signup_time'] >= start_time) &
                                       (self.MP_df['signup_time'] <= end_time)]

        if len(mixpanel_user) == 0:  # None found based on time
            return

        if len(mixpanel_user) == 1:  # One found based on time
            self.append_matching_client(mixpanel_user, client)
            return

        if len(mixpanel_user) > 1:  # More than one found based on time
            lower_premium = conversion_session['conversion_value'] - premium_margin
            upper_premium = conversion_session['conversion_value'] + premium_margin
            mixpanel_user = mixpanel_user.loc[(mixpanel_user['premium'] >= lower_premium) &
                                              (mixpanel_user['premium'] <= upper_premium)]
            if len(mixpanel_user) == 1:  # One found based on time and premium
                self.append_matching_client(mixpanel_user, client)
                return

            if len(mixpanel_user) > 1:  # More than one found based on time and premium
                source = conversion_session['source'].replace('(', '').replace(')', '')
                mixpanel_user = mixpanel_user.loc[mixpanel_user['source'].str.contains(source)]

                if len(mixpanel_user) == 1:  # One found based on time, premium and source
                    self.append_matching_client(mixpanel_user, client)
                    return

            if len(mixpanel_user) == 0:  # None found based on time and premium
                mixpanel_user = self.MP_df.loc[(self.MP_df['signup_time'] >= start_time) &
                                               (self.MP_df['signup_time'] <= end_time)]
                lower_premium = conversion_session['conversion_value'] / 0.8 - premium_margin
                upper_premium = conversion_session['conversion_value'] / 0.8 + premium_margin
                mixpanel_user = mixpanel_user.loc[(mixpanel_user['premium'] >= lower_premium) &
                                                  (mixpanel_user['premium'] <= upper_premium)]

                if len(mixpanel_user) == 1:  # One found based on time and discounted premium
                    self.append_matching_client(mixpanel_user, client)
                    return

    def append_matching_client(self, mixpanel_user, client):
        mixpanel_user.insert(0, 'client_id', client[0])
        self.converted_clients_df = self.converted_clients_df.append(mixpanel_user, ignore_index=True)

    def estimate_client_LTV(self, w_premium=1, w_nr_co_insured=-5):
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

    def process_all(self):
        self.process_individual_data()
        self.group_by_client_id()
        self.remove_post_conversion()
        self.process_mixpanel_data()
        self.create_converted_clients_df()
        self.estimate_client_LTV()


class Descriptives:
    def __init__(self, start_date, end_date, file_path_mp=None):
        self.start_date = start_date
        self.end_date = end_date
        self.data_processing = DataProcessing(self.start_date, self.end_date, file_path_mp)
        self.GA_df = None
        self.MP_df = None
        self.converted_clients_df = None
        self.read_data()

    def read_data(self):
        self.data_processing.process_all()
        self.GA_df = self.data_processing.get_GA_df()
        self.MP_df = self.data_processing.get_MP_df()
        self.converted_clients_df = self.data_processing.get_converted_clients_df()

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
        # channels = channels[::-1]
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

    def show_interesting_results_MP(self):
        self.plot_premium_age_MP()
        self.plot_age_dist_MP()
        self.plot_premium_dist_MP()

    def show_interesting_results_GA(self):
        self.plot_channel_conversion_frequency_GA()
        self.plot_path_length_GA()
        self.plot_path_duration_GA()

    def show_interesting_results_combined(self):
        self.plot_user_conversions_not_last_against_source_curve('nr_co_insured', nr_channels=3, bandwidth=0.5)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-10.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=1, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=9, hour=23, minute=59, tz='UTC')

    bq = ApiDataBigQuery(start_date=start_date, end_date=end_date)
    print(bq.get_funnel_df())

    descriptives = Descriptives(start_date, end_date, file_path_mp)
    descriptives.show_interesting_results_combined()