import pandas as pd
from DataProcessing import DataProcessing
import numpy as np
import random

class ModelDataLoader:
    def __init__(self, start_time, file_path_GA_main, file_path_GA_secondary, file_path_mp):
        self.data_processing = DataProcessing(file_path_GA_main, file_path_GA_secondary, file_path_mp)
        self.start_time = start_time
        self.GA_df = None
        self.converted_clients_df = None
        self.clients_dict = {}
        self.ch_to_idx = {}
        self.idx_to_ch = {}
        self.read_data()
        self.create_clients_dict()

    def read_data(self):
        self.data_processing.process_all(self.start_time)
        self.GA_df = self.data_processing.get_GA_df()
        self.converted_clients_df = self.data_processing.get_converted_clients_df()

    def create_clients_dict(self, use_LTV=False):
        GA_temp = self.GA_df
        self.create_idx_ch_map(GA_temp['source_medium'].unique().tolist())

        for client_id, client_df in GA_temp.groupby(level=0):
            self.process_client_df(client_id, client_df, use_LTV)

    def process_client_df(self, client_id, client_df, use_LTV):
        session_times_raw = list(client_df['timestamp'].values)
        sess_ch_names = list(client_df['source_medium'].values)
        sess_ch_idx = [self.ch_to_idx[sess_ch_name] for sess_ch_name in sess_ch_names]
        label = int(client_df['converted_eventually'][0])

        # Note that lists are not sorted on time here but in pandas df
        session_times = self.normalize_timestamps(session_times_raw)
        if use_LTV:
            if label == 1 and not self.client_converted_in_MP(client_id):
                return

        self.clients_dict[client_id] = {}
        self.clients_dict[client_id]['label'] = label
        self.clients_dict[client_id]['session_times'] = session_times
        self.clients_dict[client_id]['session_channels'] = sess_ch_idx

    def normalize_timestamps(self, session_times_raw):
        start_time = min(session_times_raw)
        session_times = []

        for session_time in session_times_raw:
            delta_days = np.timedelta64((session_time - start_time), 's').astype(float) / (3600 * 24)
            session_times.append(float(delta_days))
        return session_times

    def create_idx_ch_map(self, unique_chs):
        for idx, channel in enumerate(unique_chs):
            self.ch_to_idx[channel] = idx
            self.idx_to_ch[idx] = channel

    def client_converted_in_MP(self, client_id):
        if client_id in self.converted_clients_df['client_id'].values:
            return True
        return False

    def get_clients_dict_split(self, train_prop):
        client_ids = list(self.clients_dict.keys())
        random.shuffle(client_ids)
        nr_clients = len(client_ids)
        nr_train = round(nr_clients * train_prop)
        client_ids_train = client_ids[:nr_train]
        client_ids_test = client_ids[nr_train:]

        clients_dict_train = self.get_clients_sub_dict(client_ids_train)
        clients_dict_test = self.get_clients_sub_dict(client_ids_test)

        return clients_dict_train, clients_dict_test

    def get_clients_sub_dict(self, sub_client_ids):
        return {client_id: self.clients_dict[client_id] for client_id in sub_client_ids}

    def get_idx_to_ch_map(self):
        return self.idx_to_ch


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_GA_main = '../Data/Analytics_sample_1.csv'
    file_path_GA_secondary = '../Data/Analytics_sample_2.csv'
    file_path_mp = '../Data/Mixpanel_data_2021-02-04.csv'
    start_time_mp = pd.Timestamp(year = 2021, month = 2, day = 1, tz='UTC')

    processor = ModelDataLoader(start_time_mp, file_path_GA_main, file_path_GA_secondary, file_path_mp)

