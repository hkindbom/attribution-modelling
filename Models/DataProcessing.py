import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

class DataProcessing:
    def __init__(self, file_path_GA_main = None, file_path_GA_secondary = None, file_path_mixpanel = None, save_to_path = None):
        self.GA_df = None
        self.MP_df = None
        self.converted_clients_df = None
        self.file_path_GA_main = file_path_GA_main
        self.file_path_GA_secondary = file_path_GA_secondary
        self.file_path_mixpanel = file_path_mixpanel
        self.save_to_path = save_to_path

    def process_individual_data(self):
        df1 = pd.read_csv(self.file_path_GA_main, header=5, dtype={'cllientId': str}) ##KOM ihåg ändra header=5
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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['conversion_value'] = df['conversion_value'].str.replace('SEK','').astype(float)

        self.GA_df = df

    #def filter_journey_length(self):

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
        df = pd.read_csv(self.file_path_GA_main, header=5)
        df = df.rename(columns={'Väg till konvertering per källa': 'path',
                                'Konverteringar': 'total_conversions',
                                'Konverteringsvärde': 'total_conversion_value'})
        df['total_null'] = 0
        df['total_conversions'] = df['total_conversions'].str.replace('\s+', '').astype(int)
        df['total_conversion_value'] = df['total_conversion_value'].str.rstrip('kr').\
            str.replace(',','.').str.replace('\s+','').astype(float)
        self.GA_df = df

    def process_mixpanel_data(self, start_time = pd.Timestamp(2017,1,1), convert_to_float=True):
        df = pd.read_csv(self.file_path_mixpanel)

        # Filter out non-singups
        df = df[df['$properties.$created_at'] != 'undefined']

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

        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
        df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')

        df = df.loc[df['signup_time'] >= start_time]

        if convert_to_float:
            df['premium'] = pd.to_numeric(df['premium'], errors='coerce')
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df['zip'] = pd.to_numeric(df['zip'], errors='coerce')
            df['nr_failed_charges'] = pd.to_numeric(df['nr_failed_charges'], errors='coerce').fillna(0)
            df['nr_co_insured'] = pd.to_numeric(df['nr_co_insured'], errors='coerce').fillna(0)

        self.MP_df = df

    def create_converted_clients_df(self, minute_margin = 1, premium_margin = 10):
        #self.merged_df[list(self.mixpanel_df.columns)] = pd.DataFrame([[np.nan]*len(self.mixpanel_df.columns)])
        self.converted_clients_df = pd.DataFrame(columns=['client_id'] + list(self.MP_df.columns))

        conversion_sessions_df = self.GA_df.loc[self.GA_df['conversion'] == 1]
        for client, conversion_session in conversion_sessions_df.iterrows():
            self.match_client(client, conversion_session, minute_margin, premium_margin)
        print(self.converted_clients_df.head(10))


    def match_client(self, client, conversion_session, minute_margin, premium_margin):
        conversion_time = pd.to_datetime(conversion_session['timestamp'].strftime('%Y-%m-%d %H:%M:%S'))
        starttime = conversion_time - timedelta(minutes = minute_margin)
        endtime = conversion_time + timedelta(minutes = minute_margin)

        mixpanel_user = self.MP_df.loc[(self.MP_df['signup_time'] > starttime) &
                                       (self.MP_df['signup_time'] < endtime)]

        if len(mixpanel_user) == 0:
            return

        if len(mixpanel_user) == 1:
            self.append_matching_client(mixpanel_user, client)
            return

        if len(mixpanel_user) > 1:  ## If multiple measures
            lower_premium = conversion_session['conversion_value'] - premium_margin
            upper_premium = conversion_session['conversion_value'] + premium_margin
            mixpanel_user = mixpanel_user.loc[(mixpanel_user['premium'] > lower_premium) &
                                              (mixpanel_user['premium'] < upper_premium)]
            if len(mixpanel_user) == 1:
                self.append_matching_client(mixpanel_user, client)

    def append_matching_client(self, mixpanel_user, client):
        mixpanel_user.insert(0, 'client_id', client[0])
        self.converted_clients_df = self.converted_clients_df.append(mixpanel_user, ignore_index=True)


    def save_to_csv(self):
        self.GA_df.to_csv(self.save_to_path, sep=',')

    def get_GA_df(self):
        return self.GA_df

    def get_MP_df(self):
        return self.MP_df

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

    #def number_conversions(self):
        ##

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

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_GA_main = '../Data/Analytics_sample_1.csv'
    file_path_GA_secondary = '../Data/Analytics_sample_2.csv'
    file_path_mp = '../Data/Mixpanel_data_2021-02-03.csv'

    #data_processing.save_to_csv()
    #df = data_processing.get_mixpanel_df()

    descriptives = Descriptives(pd.Timestamp(2021,2,1), file_path_GA_main, file_path_GA_secondary, file_path_mp)
    descriptives.show_interesting_results_MP()

## exploratory data analysis class (Descriptives); sum number of conversions, total conversion value...
## (make function that gets client, returns dataframe with unique user_ids)
## chart (e.g. PyChart) with percentage conversions etc.
## ranked channels
## distribution of path length

## integration with MarkovModel file, create DataProcessing object, return df instead of read csv