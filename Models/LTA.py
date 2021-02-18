from ModelDataLoader import ModelDataLoader

class LTA:
    def __init__(self, start_date, end_date, file_path_mp, nr_top_ch, train_prop=0.8, ratio_maj_min_class=1):
        self.data_loader = ModelDataLoader(start_date, end_date, file_path_mp, nr_top_ch, ratio_maj_min_class)
        self.clients_data_train = {}
        self.clients_data_test = {}
        self.train_prop = train_prop

    def load_train_test_data(self):
        self.clients_data_train, self.clients_data_test = self.data_loader.get_clients_dict_split(self.train_prop)

    def train(self):
        self.load_train_test_data()
        client_ids_train = list(self.clients_data_train.keys())
