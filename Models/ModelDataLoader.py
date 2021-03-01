import pandas as pd
from DataProcessing import DataProcessing
from Simulator import Simulator
import numpy as np
import random
from collections import Counter

class ModelDataLoader:
    def __init__(self, start_date_data, end_date_data, start_data_cohort, end_data_cohort,
                 file_path_mp, nr_top_ch=1000, ratio_maj_min_class=1, simulate=False, cohort_size=100, sim_time=100):
        self.data_processing = DataProcessing(start_date_data, end_date_data, start_data_cohort,
                                              end_data_cohort, file_path_mp, nr_top_ch=nr_top_ch,
                                              ratio_maj_min_class=ratio_maj_min_class)
        self.GA_df = None
        self.converted_clients_df = None
        self.clients_dict = {}
        self.ch_to_idx = {}
        self.idx_to_ch = {}
        self.simulate = simulate
        self.cohort_size = cohort_size
        self.sim_time = sim_time
        self.load_data()

    def load_data(self):
        if self.simulate:
            self.load_sim_data()
            return
        self.load_real_data()

    def load_sim_data(self):
        sim = Simulator(self.cohort_size, self.sim_time)
        sim.run_simulation()
        self.clients_dict = sim.get_data_dict_format()
        self.ch_to_idx, self.idx_to_ch = sim.get_ch_idx_maps()

    def load_real_data(self):
        self.data_processing.process_all()
        self.GA_df = self.data_processing.get_GA_df()
        self.converted_clients_df = self.data_processing.get_converted_clients_df()
        self.create_clients_dict()

    def create_clients_dict(self, use_LTV=False):
        GA_temp = self.GA_df
        self.create_idx_ch_map(GA_temp['source_medium'].unique().tolist())
        for client_id, client_df in GA_temp.groupby(level=0):
            self.process_client_df(client_id, client_df, use_LTV)

    def process_client_df(self, client_id, client_df, use_LTV):
        session_times_raw = list(client_df['timestamp'].values)
        sess_ch_names = list(client_df['source_medium'].values)
        sess_ch_cost = list(client_df['cost'].values)
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
        self.clients_dict[client_id]['cost'] = sess_ch_cost

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

    def get_all_seq_lists_and_labels(self):
        seq_lists = []
        labels = []
        for client_id in self.clients_dict:
            session_channels = self.clients_dict[client_id]['session_channels']
            labels.append(self.clients_dict[client_id]['label'])
            seq_lists.append(session_channels)
        return seq_lists, labels

    def get_theo_max_accuracy(self):
        counter_pos = Counter()
        counter_tot = Counter()
        for client_id in self.clients_dict:
            session_channels = self.clients_dict[client_id]['session_channels']
            key = self.get_str_key_from_chs(session_channels)
            counter_pos[key] += self.clients_dict[client_id]['label']
            counter_tot[key] += 1
        return self.calc_theo_acc_max(counter_pos, counter_tot)

    def calc_theo_acc_max(self, counter_pos, counter_tot):
        nr_errors = 0
        tot_samples = len(list(counter_tot.elements()))
        for seq_key in list(counter_tot):
            nr_errors += min(counter_pos[seq_key], counter_tot[seq_key]-counter_pos[seq_key])
        return 1-(nr_errors / tot_samples)

    def get_str_key_from_chs(self, channels):
        str_channels = [str(ch) for ch in channels]
        return '_'.join(str_channels)

    def get_feature_matrix_split(self, train_prop, use_time=False, use_LTV=False):
        nr_channels = len(self.ch_to_idx)
        nr_clients = len(self.clients_dict)
        labels = np.empty(nr_clients)
        feature_matrix = np.zeros((nr_clients, nr_channels))

        for idx, client_id in enumerate(list(self.clients_dict.keys())):
            label = self.clients_dict[client_id]['label']
            if use_LTV:
                pass
            else:
                labels[idx] = label
            feature_matrix[idx, self.clients_dict[client_id]['session_channels']] = 1
            if use_time:
                times = np.array(self.clients_dict[client_id]['session_times'])
                times_on_matrix_format = np.zeros(nr_channels)
                times_on_matrix_format[self.clients_dict[client_id]['session_channels']] = times
                feature_matrix[idx] += times_on_matrix_format

        random_indices = np.random.permutation(len(labels))
        randomized_matrix = feature_matrix[random_indices]
        randomized_labels = labels[random_indices]

        nr_train = round(nr_clients * train_prop)
        x_train = randomized_matrix[:nr_train]
        x_test = randomized_matrix[nr_train:]
        y_train = randomized_labels[:nr_train]
        y_test = randomized_labels[nr_train:]
        return x_train, y_train, x_test, y_test

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

    def get_ch_to_idx_map(self):
        return self.ch_to_idx

    def get_GA_df(self):
        return self.GA_df

    def get_converted_clients_df(self):
        return self.converted_clients_df


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-22.csv'
    start_date_data = pd.Timestamp(year=2021, month=2, day=2, hour=0, minute=0, tz='UTC')
    end_date_data = pd.Timestamp(year=2021, month=2, day=21, hour=23, minute=59, tz='UTC')

    start_date_cohort = pd.Timestamp(year=2021, month=2, day=5, hour=0, minute=0, tz='UTC')
    end_date_cohort = pd.Timestamp(year=2021, month=2, day=13, hour=23, minute=59, tz='UTC')

    processor = ModelDataLoader(start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mp,
                                simulate=True)
    processor.get_feature_matrix_split(0.8)
