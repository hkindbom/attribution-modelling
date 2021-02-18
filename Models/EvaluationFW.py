import pandas as pd
from SP import SP
from LR import LR

class Evaluation:
    def __init__(self, GA_df, converted_clients_df, total_budget, attributions, ch_to_idx_map):
        self.GA_df = GA_df
        self.converted_clients_df = converted_clients_df
        self.total_budget = total_budget
        self.attributions = attributions
        self.ch_to_idx_map = ch_to_idx_map
        self.channels_roi = {}
        self.channels_budgets = {}

    def calculate_channels_roi(self):
        conversion_paths_df = self.GA_df.loc[self.GA_df['converted_eventually'] == 1]
        channels_spend = self.GA_df.groupby(['source_medium']).agg('sum')['cost']
        channels_spend = channels_spend[channels_spend > 0]

        for channel_name, channel_spend in channels_spend.iteritems():
            conv_paths_w_channel_df = conversion_paths_df[conversion_paths_df['source_medium'] == channel_name]
            ch_idx = self.ch_to_idx_map[channel_name]
            attribution = self.attributions[ch_idx]
            _return = 0
            for client, path_w_channel in conv_paths_w_channel_df.groupby(level=0):
                nr_channel_occurences_touchpts = len(path_w_channel)
                conversion_value = path_w_channel['conversion_value'].max()
                _return += nr_channel_occurences_touchpts * attribution * conversion_value
            self.channels_roi[channel_name] = _return / channel_spend

    def calculate_channels_budgets(self):
        for channel_name, channel_roi in self.channels_roi.items():
            roi_proportion = channel_roi / sum(self.channels_roi.values())
            self.channels_budgets[channel_name] = roi_proportion * self.total_budget

    def back_evaluation(self):
        client_blacklist = []
        total_nr_conversions, total_conversion_value, total_cost, total_ltv = 0, 0, 0, 0
        for client_id, path in self.GA_df.groupby(level=0):
            for _, session in path.iterrows():
                if client_id not in client_blacklist:
                    cost = session['cost']
                    channel = session['source_medium']
                    budget_allows = True
                    if channel in self.channels_budgets.keys():
                        if self.channels_budgets[channel] > cost:
                            self.channels_budgets[channel] -= cost
                        else:
                            budget_allows = False
                    if budget_allows:
                        total_cost += cost
                        if session['conversion']:
                            total_nr_conversions += session['conversion']
                            total_conversion_value += session['conversion_value']
                            total_ltv += self.get_client_LTV(client_id)
                    else:
                        client_blacklist.append(client_id)
        print('Total conversions:', total_nr_conversions)
        print('Total conversion value:', total_conversion_value)
        print('Total LTV:', total_ltv)
        print('Total cost spent:', total_cost, 'out of', self.total_budget)

    def get_client_LTV(self, client_id):
        converted_client_df = self.converted_clients_df.loc[self.converted_clients_df['client_id'] == client_id]
        if not converted_client_df.empty:
            return converted_client_df.iloc[0]['LTV']
        return 0

    def evaluate(self):
        self.calculate_channels_roi()
        self.calculate_channels_budgets()
        self.back_evaluation()


if __name__ == '__main__':
    file_path_mp = '../Data/Mixpanel_data_2021-02-17.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=2, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=16, hour=23, minute=59, tz='UTC')

    train_prop = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 1
    use_time = True
    total_budget = 1000

    model = LR(start_date, end_date, file_path_mp, nr_top_ch, use_time, train_prop, ratio_maj_min_class)
    #model = SP(start_date, end_date, file_path_mp, nr_top_ch, train_prop, ratio_maj_min_class)

    model.train()
    attributions = model.get_attributions()
    GA_df = model.get_GA_df()
    converted_clients_df = model.get_converted_clients_df()
    ch_to_idx_map = model.get_ch_to_idx_map()

    evaluation = Evaluation(GA_df, converted_clients_df, total_budget, attributions, ch_to_idx_map)
    evaluation.evaluate()