import pandas as pd
import numpy as np
import calendar
from datetime import timedelta
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account
from google.cloud import bigquery, bigquery_storage

class ApiDataBigQuery:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        self.funnel_df = pd.DataFrame()
        self.fetch_BQ()

    def fetch_BQ(self):
        credentials = service_account.Credentials.from_service_account_file('../API/BQ_api.json')
        bqclient = bigquery.Client(credentials=credentials, project=credentials.project_id)
        bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials)

        start_date_temp = self.start_date.replace(day=1)
        month_intervals = pd.date_range(start_date_temp.date(), self.end_date.date(), freq='MS', normalize=True).tolist()
        for month_period in month_intervals:
            if month_period == month_intervals[0]:
                query_start_date = f"{self.start_date.year}, {self.start_date.month}, {self.start_date.day}"

            else:
                query_start_date = f"{month_period.year}, {month_period.month}, {1}"

            if month_period == month_intervals[-1]:
                query_end_date = f"{self.end_date.year}, {self.end_date.month}, {self.end_date.day}"

            else:
                query_end_date = f"{month_period.year}, {month_period.month}, " \
                                 f"{calendar.monthrange(month_period.year, month_period.month)[1]}"

            query_string = f"SELECT Date, Traffic_source, Data_Source_type, Cost, Clicks, Impressions \
                           FROM funnel-integration.Marketing_Spend.marketing_spend_monthly_{month_period.year}_{str(month_period.month).zfill(2)} \
                           WHERE Date >= DATE ({query_start_date}) \
                           AND Date <= DATE ({query_end_date}) \
                           AND (Campaign_name__TikTok NOT LIKE '%no%' OR Campaign_name__TikTok IS NULL)"

            self.funnel_df = self.funnel_df.append(bqclient.query(query_string).result()
                                                   .to_dataframe(bqstorage_client=bqstorageclient))

        self.add_cost_per_click()
        print('Read ', len(self.funnel_df), ' datapoints from BigQuery Funnel (in ',
              len(month_intervals), ' querys)')

    def add_cost_per_click(self):
        self.funnel_df['cpc'] = self.funnel_df['Cost'] / self.funnel_df['Clicks']
        self.funnel_df.loc[~np.isfinite(self.funnel_df['cpc']), 'cpc'] = np.nan  # Handle div by 0

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
        print('Read ', len(self.GA_api_df), ' datapoints from Google Analytics before processing')

    def get_GA_df(self):
        return self.GA_api_df

    def initialize_api(self):
        self.initialize_analyticsreporting()
        self.create_report_df()


class DataProcessing:
    def __init__(self, start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mixpanel=None,
                 file_path_GA_aggregated=None, save_to_path=None, nr_top_ch=10000, ratio_maj_min_class=None):
        self.start_date_data = start_date_data
        self.end_date_data = end_date_data
        self.start_date_cohort = start_date_cohort
        self.end_date_cohort = end_date_cohort
        self.nr_top_ch = nr_top_ch
        self.ratio_maj_min_class = ratio_maj_min_class
        self.GA_df = None
        self.MP_df = None
        self.converted_clients_df = None
        self.GA_aggregated_df = None
        self.funnel_df = None
        self.file_path_mixpanel = file_path_mixpanel
        self.file_path_GA_aggregated = file_path_GA_aggregated
        self.save_to_path = save_to_path


    def process_individual_data(self):
        GA_api = ApiDataGA(self.start_date_data, self.end_date_data)
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

        print('Number of unique sources in GA before filter: ', len(GA_api_df['source'].unique()))
        self.GA_df = GA_api_df

    def drop_uncommon_channels(self):
        source_counts = self.GA_df['source_medium'].value_counts()
        if len(source_counts) <= self.nr_top_ch:
            return
        self.GA_df = self.GA_df.groupby('source_medium').filter(lambda source: len(source) >= source_counts[self.nr_top_ch-1])

    def drop_duplicate_sessions(self):
        self.GA_df.sort_values(by=['client_id', 'timestamp'], ascending=True)
        self.GA_df = self.GA_df.drop_duplicates(subset=['client_id', 'session_id'], keep='last')

    def filter_cohort_sessions(self):
        cohort_sessions_df = self.GA_df.loc[(self.GA_df['timestamp'] >= self.start_date_cohort) &
                                            (self.GA_df['timestamp'] <= self.end_date_cohort)]
        pre_cohort_df = self.GA_df.loc[self.GA_df['timestamp'] < self.start_date_cohort]
        self.GA_df = cohort_sessions_df[~cohort_sessions_df['client_id'].isin(pre_cohort_df['client_id'])]

    def balance_classes_GA(self):
        if self.ratio_maj_min_class is None:
            return
        GA_temp = self.GA_df
        class_counts = GA_temp['conversion'].value_counts()
        major_label = class_counts.index.tolist()[0]
        minor_label = class_counts.index.tolist()[1]

        GA_major_downsampled = GA_temp.query('converted_eventually == ' +
                                             str(major_label)).sample(round(class_counts[1] * self.ratio_maj_min_class))
        GA_minority = GA_temp[GA_temp['converted_eventually'] == minor_label]
        self.GA_df = GA_minority.append(GA_major_downsampled).sort_index()

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
        self.GA_df = pd.concat([df, sessions_to_delete]).drop_duplicates(keep=False)

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

        df = df.loc[(df['signup_time'] >= self.start_date_data) &
                    (df['signup_time'] <= self.end_date_data)]
        df = df.loc[df['market'] == market]

        if convert_to_float:
            df['premium'] = pd.to_numeric(df['premium'], errors='coerce')
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['zip'] = pd.to_numeric(df['zip'], errors='coerce')
            df['nr_failed_charges'] = pd.to_numeric(df['nr_failed_charges'], errors='coerce').fillna(0)
            df['nr_co_insured'] = pd.to_numeric(df['nr_co_insured'], errors='coerce').fillna(0)

        self.MP_df = df
        print('Read ', len(self.MP_df), ' datapoints from Mixpanel')

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

    def assign_cost(self, free_mediums):
        self.GA_df['cost'] = 0
        paid_click_sessions_df = self.GA_df.loc[~self.GA_df['source_medium'].str.contains('|'.join(free_mediums))]
        for cust_id, session in paid_click_sessions_df.iterrows():
            marketing_spend_series = self.funnel_df.loc[(self.funnel_df['Date'] == session['timestamp'].date()) &
                                                   (self.funnel_df['Traffic_source'].str.lower() == session['source'])]
            if not marketing_spend_series.empty:
                self.GA_df.loc[cust_id, 'cost'] = marketing_spend_series.iloc[0]['cpc']

    def process_bq_funnel(self):
        bq_processor = ApiDataBigQuery(self.start_date_data, self.end_date_data)
        self.funnel_df = bq_processor.get_funnel_df()

    def save_to_csv(self):
        self.GA_df.to_csv(self.save_to_path, sep=',')

    def get_GA_df(self):
        return self.GA_df

    def get_MP_df(self):
        return self.MP_df

    def get_GA_aggr_df(self):
        return self.GA_aggregated_df

    def get_funnel_df(self):
        return self.funnel_df

    def get_converted_clients_df(self):
        return self.converted_clients_df

    def get_client_mixpanel_info(self, client):
        return self.converted_clients_df.loc[self.converted_clients_df['client_id'] == client]

    def process_all(self):
        self.process_bq_funnel()
        self.process_individual_data()
        self.drop_duplicate_sessions()
        self.filter_cohort_sessions()
        self.drop_uncommon_channels()
        self.group_by_client_id()
        self.remove_post_conversion()
        self.balance_classes_GA()

        self.process_mixpanel_data()
        self.create_converted_clients_df()
        self.estimate_client_LTV()
        self.assign_cost(['organic'])


## Temporary, to be deleted:
if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-22.csv'
    start_date_data = pd.Timestamp(year=2021, month=2, day=2, hour=0, minute=0, tz='UTC')
    end_date_data = pd.Timestamp(year=2021, month=2, day=21, hour=23, minute=59, tz='UTC')

    start_date_cohort = pd.Timestamp(year=2021, month=2, day=5, hour=0, minute=0, tz='UTC')
    end_date_cohort = pd.Timestamp(year=2021, month=2, day=13, hour=23, minute=59, tz='UTC')

    data_processing = DataProcessing(start_date_data, end_date_data, start_date_cohort, end_date_cohort,
                                     file_path_mp, nr_top_ch=1000, ratio_maj_min_class=1)
    data_processing.process_all()
    GA_df = data_processing.get_GA_df()