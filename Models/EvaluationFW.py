import pandas as pd
from SP import SP
from LR import LR
from DataProcessing import DataProcessing

class Evaluation:
    def __init__(self, start_date, end_date, file_path_mp, nr_top_ch, use_time, train_prop, ratio_maj_min_class,
                 total_budget):
        self.lr_model = LR(start_date, end_date, file_path_mp, nr_top_ch, use_time, train_prop, ratio_maj_min_class)
        self.sp_model = SP(start_date, end_date, file_path_mp, nr_top_ch, train_prop, ratio_maj_min_class)
        self.data_processing = DataProcessing(start_date, end_date, file_path_mp, nr_top_ch=nr_top_ch,
                                              ratio_maj_min_class = ratio_maj_min_class)
        self.data_processing.process_all()
        self.total_budget = total_budget
        #self.model =

    def train_lr_model(self):
        self.lr_model.train()

    def train_sp_model(self):
        self.sp_model.train()

    def calculate_roi(self):
        GA_df = self.data_processing.get_GA_df()
        funnel_df = self.data_processing.get_funnel_df()

    def calculate_channel_budgets(self):
        pass

    def back_evaluation(self):
        GA_df = self.data_processing.get_GA_df()
        print(GA_df.head(10))
        conversion_blacklist = []
        total_nr_conversions, total_conversion_value, total_cost = 0, 0, 0
       # for _ in events:
       #     if sequence not in conversion_blacklist:
       #         if


if __name__ == '__main__':
    file_path_mp = '../Data/Mixpanel_data_2021-02-11.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=10, hour=23, minute=59, tz='UTC')

    train_prop = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 1
    use_time = True

    evaluation = Evaluation(start_date, end_date, file_path_mp, nr_top_ch, use_time, train_prop, ratio_maj_min_class)
    evaluation.back_evaluation()