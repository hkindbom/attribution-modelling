import pandas as pd
from SP import SP
from LR import LR

class Evaluation:
    def __init__(self, GA_df, total_budget, attributions, idx_to_ch_map, ch_to_idx_map):
        self.GA_df = GA_df
        self.total_budget = total_budget
        self.attributions = attributions
        self.idx_to_ch_map = idx_to_ch_map
        self.ch_to_idx_map = ch_to_idx_map
        self.channels_roi = {}
        self.channels_budgets = {}

    def calculate_channels_roi(self):
        conversion_paths = self.GA_df.loc[self.GA_df['converted_eventually'] == 1]
        channels_spend = self.GA_df.groupby(['source_medium']).agg('sum')['cost']
        channels_spend = channels_spend[channels_spend > 0]

        for channel_name, channel_spend in channels_spend.iteritems():
            conv_paths_w_channel_df = conversion_paths[conversion_paths['source_medium'] == channel_name]
            ch_idx = self.ch_to_idx_map[channel_name]
            attribution = attributions[ch_idx]
            _return = 0
            for client, path in conv_paths_w_channel_df.groupby(level=0):
                nr_channel_touchpts = len(path)
                conversion_value = path['conversion_value'].max()
                _return += nr_channel_touchpts * attribution * conversion_value
            self.channels_roi[channel_name] = _return / channel_spend

    def calculate_channels_budgets(self):
        for channel_name, channel_roi in self.channels_roi:
            roi_proportion = channel_roi / sum(self.channels_roi.values())
            self.channels_budgets[channel_name] = roi_proportion * self.total_budget

    def back_evaluation(self):
        client_blacklist = []
        total_nr_conversions, total_conversion_value, total_cost = 0, 0, 0
        for client_id, path in self.GA_df.groupby(level=0):
            if sequence not in conversion_blacklist:
                #if


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-11.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=10, hour=23, minute=59, tz='UTC')

    train_prop = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 1
    use_time = True
    total_budget = 100000

    model = LR(start_date, end_date, file_path_mp, nr_top_ch, use_time, train_prop, ratio_maj_min_class)
    model.train()
    attributions = model.get_attributions()
    print('attributions', attributions)
    GA_df = model.get_GA_df()
    idx_to_ch_map = model.get_idx_to_ch_map()
    ch_to_idx_map = model.get_ch_to_idx_map()

    evaluation = Evaluation(GA_df, total_budget, attributions, idx_to_ch_map, ch_to_idx_map)
    evaluation.calculate_channels_roi()