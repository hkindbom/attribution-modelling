import pandas as pd
from DataProcessing import DataProcessing

class ProcessToNumeric:
    def __init__(self, start_time, file_path_GA_main, file_path_GA_secondary, file_path_mp):
        self.data_processing = DataProcessing(file_path_GA_main, file_path_GA_secondary, file_path_mp)
        self.start_time = start_time
        self.GA_df = None
        self.MP_df = None
        self.converted_clients_df = None
        self.clients_dict = {}
        self.ch_to_idx = {}
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

    def create_clients_dict(self):
        GA_temp = self.GA_df
        self.create_ch_to_idx_map(GA_temp['source_medium'].unique().tolist())

        clients_dict_raw = GA_temp.groupby(level=0).apply(lambda GA_temp: GA_temp.xs(GA_temp.name).to_dict()).to_dict()
        self.process_raw_clients_dict(clients_dict_raw)

    def create_ch_to_idx_map(self, unique_chs):
        for idx, channel in enumerate(unique_chs):
            self.ch_to_idx[channel] = idx

    def process_raw_clients_dict(self, clients_dict_raw):
        for client_id in clients_dict_raw:
            client_dict = clients_dict_raw[client_id]
            self.process_client_dict(client_dict, client_id)

    # Note that lists are not sorted on time
    def process_client_dict(self, client_dict, client_id):
        client_sessions = list(client_dict['sessions'].keys())

        touch_times = list(client_dict['timestamp'].values())
        start_time = min(touch_times)
        session_times = []
        session_channels = []

        for client_session in client_sessions:
            client_session_time = client_dict['timestamp'][client_session]
            session_times.append((client_session_time - start_time).total_seconds()/(3600*24))

            ch_idx = self.ch_to_idx[client_dict['source_medium'][client_session]]
            session_channels.append(ch_idx)

        label = client_dict['converted_eventually'][client_sessions[0]]
        if label == 1:
            if not self.client_converted_in_MP(client_id):
                label = 0

        self.clients_dict[client_id] = {}
        self.clients_dict[client_id]['label'] = label
        self.clients_dict[client_id]['session_times'] = session_times
        self.clients_dict[client_id]['session_channels'] = session_channels

    def client_converted_in_MP(self, client_id):
        if client_id in self.converted_clients_df['client_id'].values:
            return True
        return False

    def get_clients_dict(self):
        return self.clients_dict


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_GA_main = '../Data/Analytics_sample_1.csv'
    file_path_GA_secondary = '../Data/Analytics_sample_2.csv'
    file_path_mp = '../Data/Mixpanel_data_2021-02-04.csv'
    start_time_mp = pd.Timestamp(year = 2021, month = 2, day = 1, tz='UTC')

    processor = ProcessToNumeric(start_time_mp, file_path_GA_main, file_path_GA_secondary, file_path_mp)
    processor.create_clients_dict()
