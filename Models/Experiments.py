import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ModelDataLoader import ModelDataLoader
from EvaluationFW import Evaluation
from SP import SP
from LTA import LTA
from LR import LR
from LSTM import LSTM

class Experiments:

    def __init__(self, start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mp, nr_top_ch,
                 train_prop, ratio_maj_min_class, use_time, simulate, cohort_size, sim_time, epochs, batch_size,
                 learning_rate, ctrl_var, ctrl_var_value, eval_fw, total_budget, custom_attr_eval):

        self.data_loader = ModelDataLoader(start_date_data, end_date_data, start_date_cohort, end_date_cohort,
                                           file_path_mp, nr_top_ch, ratio_maj_min_class, simulate, cohort_size,
                                           sim_time, ctrl_var, ctrl_var_value, eval_fw)
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.SP_model = None
        self.LTA_model = None
        self.LR_model = None
        self.LSTM_model = None
        self.use_time = use_time
        self.simulate = simulate
        self.idx_to_ch = self.data_loader.get_idx_to_ch_map()
        self.ch_to_idx = self.data_loader.get_ch_to_idx_map()
        self.attributions = {}
        self.attributions_std = {}
        self.train_prop = train_prop
        self.nr_top_ch = nr_top_ch
        self.model_stats = {}
        self.eval_fw = eval_fw
        self.custom_attr_eval = custom_attr_eval
        self.total_budget = total_budget

    def init_models(self):
        self.SP_model = SP()
        self.LTA_model = LTA()
        self.LR_model = LR()
        self.LSTM_model = LSTM(self.epochs, self.batch_size, self.learning_rate)

    def cv(self, tot_budget, nr_splits=5):
        train_prop = 1
        clients_data, _ = self.data_loader.get_clients_dict_split(train_prop)
        x, y, _, _ = self.data_loader.get_feature_matrix_split(train_prop, self.use_time)
        seq_lists, labels, _, _ = self.data_loader.get_seq_lists_split(train_prop)

        tot_nr_samples = len(clients_data)
        nr_samples_test = tot_nr_samples // nr_splits
        for split_idx in range(nr_splits):
            self.init_models()
            test_start_idx = int(nr_samples_test * split_idx)
            test_end_idx = int(nr_samples_test * (split_idx + 1))
            clients_data_train, clients_data_test = self.get_train_test_dicts(clients_data, test_start_idx, test_end_idx)
            x_train, y_train, x_test, y_test = self.get_train_test_arr(x, y, test_start_idx, test_end_idx)
            seq_lists_train, labels_train, seq_lists_test, labels_test = self.get_train_test_list(seq_lists, labels, test_start_idx, test_end_idx)
            self.load_models(clients_data_train, clients_data_test, x_train, y_train, x_test, y_test,
                    seq_lists_train, labels_train, seq_lists_test, labels_test)

            self.train_all()
            self.collect_models_pred_stats()
            self.collect_models_attr(nr_splits, split_idx)
        self.show_cv_results()
        self.profit_eval(tot_budget)

    def show_cv_results(self):
        models_res = self.calc_mean_and_std()
        self.show_pred_res(models_res, cv=True)
        self.plot_attributions(print_sum_attr=False, cv=True)

    def calc_mean_and_std(self):
        models_res = self.calc_mean_pred()
        self.calc_mean_and_std_attr()
        return models_res

    def calc_mean_pred(self):
        models_res = []
        for model_name in self.model_stats:
            model_stats_filt = self.model_stats[model_name].copy()
            model_res_means, model_stats_filt = self.calc_metrics(model_stats_filt)
            for model_stat in model_stats_filt:
                stat_list = model_stats_filt[model_stat]
                model_res_means[model_stat] = sum(stat_list) / len(stat_list)
            model_res_means['model'] = model_name
            models_res.append(model_res_means)
        return models_res

    def calc_metrics(self, model_stats_filt):
        model_res_means = {}
        model_res_means['precision'] = (np.array(model_stats_filt['tp']) / (np.array(model_stats_filt['tp']) + np.array(model_stats_filt['fp']))).mean()
        model_res_means['recall'] = (np.array(model_stats_filt['tp']) / (np.array(model_stats_filt['tp']) + np.array(model_stats_filt['fn']))).mean()

        model_res_means['F1'] = 2 * model_res_means['precision'] * model_res_means['recall'] / (
                    model_res_means['precision'] + model_res_means['recall'])

        model_res_means['accuracy'] = ((np.array(model_stats_filt['tp']) + np.array(model_stats_filt['tn'])) / (
                    np.array(model_stats_filt['tn']) + np.array(model_stats_filt['tp']) + np.array(model_stats_filt['fp']) + np.array(model_stats_filt['fn']))).mean()

        model_stats_filt.pop('tn')
        model_stats_filt.pop('fp')
        model_stats_filt.pop('fn')
        model_stats_filt.pop('tp')
        model_stats_filt.pop('model')
        model_stats_filt.pop('attributions')
        return model_res_means, model_stats_filt

    def calc_mean_and_std_attr(self):
        for model_name in self.model_stats:
            self.attributions[model_name] = self.model_stats[model_name]['attributions'].mean(axis=0).tolist()
            self.attributions_std[model_name] = self.model_stats[model_name]['attributions'].std(axis=0).tolist()

    def collect_models_attr(self, nr_splits, split_idx):
        models_attr_dict = self.load_attributions(output=True)
        for model_name in models_attr_dict:
            if 'attributions' not in self.model_stats[model_name]:
                self.model_stats[model_name]['attributions'] = np.zeros((nr_splits, len(self.ch_to_idx)))
            self.model_stats[model_name]['attributions'][split_idx] = np.array(models_attr_dict[model_name])

    def collect_models_pred_stats(self):
        models_res = self.validate_pred(output=True)
        for model_res in models_res:
            self.collect_model_pred_stats(model_res['model'], model_res)

    def collect_model_pred_stats(self, model_name, model_res):
        if model_name not in self.model_stats:
            self.model_stats[model_name] = {}
            for model_stat in model_res:
                self.model_stats[model_name][model_stat] = [model_res[model_stat]]
            return
        self.add_model_stats(model_name, model_res)

    def add_model_stats(self, model_name, model_stats):
        for model_stat in model_stats:
            self.model_stats[model_name][model_stat].append(model_stats[model_stat])

    def get_train_test_arr(self, x, y, test_start_idx, test_end_idx):
        if test_start_idx == 0:
            x_train = x[test_end_idx:]
            y_train = y[test_end_idx:]
        elif test_end_idx == len(x)-1:
            x_train = x[:test_start_idx]
            y_train = y[:test_start_idx]
        else:
            x_train = np.concatenate((x[:test_start_idx], x[test_end_idx:]), axis=0)
            y_train = np.concatenate((y[:test_start_idx], y[test_end_idx:]), axis=0)
        x_test = x[test_start_idx:test_end_idx]
        y_test = y[test_start_idx:test_end_idx]
        return x_train, y_train, x_test, y_test

    def get_train_test_list(self, seq_lists, labels, test_start_idx, test_end_idx):
        seq_lists_train = seq_lists[:test_start_idx] + seq_lists[test_end_idx:]
        labels_train = labels[:test_start_idx] + labels[test_end_idx:]
        seq_lists_test = seq_lists[test_start_idx:test_end_idx]
        labels_test = labels[test_start_idx:test_end_idx]
        return seq_lists_train, labels_train, seq_lists_test, labels_test

    def get_train_test_dicts(self, clients_data, test_start_idx, test_end_idx):
        all_client_ids = list(clients_data.keys())
        train_client_ids = all_client_ids[:test_start_idx] + all_client_ids[test_end_idx:]
        test_client_ids = all_client_ids[test_start_idx:test_end_idx]
        clients_data_train = {client_id: clients_data[client_id] for client_id in train_client_ids}
        clients_data_test = {client_id: clients_data[client_id] for client_id in test_client_ids}
        return clients_data_train, clients_data_test

    def load_data(self):
        clients_data_train, clients_data_test = self.data_loader.get_clients_dict_split(self.train_prop)
        x_train, y_train, x_test, y_test = self.data_loader.get_feature_matrix_split(self.train_prop, self.use_time)
        seq_lists_train, labels_train, seq_lists_test, labels_test = self.data_loader.get_seq_lists_split(self.train_prop)

        self.load_models(clients_data_train, clients_data_test, x_train, y_train, x_test, y_test,
                        seq_lists_train, labels_train, seq_lists_test, labels_test)

    def load_models(self, clients_data_train, clients_data_test, x_train, y_train, x_test, y_test,
                    seq_lists_train, labels_train, seq_lists_test, labels_test):
        self.SP_model.load_train_test_data(clients_data_train, clients_data_test)
        self.LTA_model.load_train_test_data(clients_data_train, clients_data_test)
        self.LR_model.load_train_test_data(x_train, y_train, x_test, y_test)
        self.LSTM_model.load_data(seq_lists_train, labels_train, seq_lists_test, labels_test)

    def train_all(self):
        self.SP_model.train()
        self.LTA_model.train()
        self.LR_model.train()
        self.LSTM_model.train()

    def validate_pred(self, output=False):
        LTA_res = self.LTA_model.get_results()
        LR_res = self.LR_model.get_results()
        SP_res = self.SP_model.get_results()
        LSTM_res = self.LSTM_model.get_results()
        LTA_res['model'] = 'LTA'
        LR_res['model'] = 'LR'
        SP_res['model'] = 'SP'
        LSTM_res['model'] = 'LSTM'
        models_res = [LTA_res, LR_res, SP_res, LSTM_res]
        if output:
            return models_res
        self.show_pred_res(models_res)
        if self.eval_fw:
            self.add_custom_attr()
            self.profit_eval()

    def show_pred_res(self, models_res, cv=False):
        results_df = pd.DataFrame()
        for model_res in models_res:
            results_df = results_df.append(model_res, ignore_index=True)

        if not cv:
            results_df['precision'] = results_df['tp'] / (results_df['tp'] + results_df['fp'])
            results_df['recall'] = results_df['tp'] / (results_df['tp'] + results_df['fn'])
            results_df['F1'] = 2 * results_df['precision'] * results_df['recall'] / (results_df['precision'] + results_df['recall'])
            results_df['accuracy'] = (results_df['tp'] + results_df['tn']) / (results_df['tn'] + results_df['tp'] + results_df['fp'] + results_df['fn'])

        print('Theoretical max accuracy on all data is: ', self.data_loader.get_theo_max_accuracy())
        print(results_df)

    def load_attributions(self, output=False):
        SP_attr = self.SP_model.get_normalized_attributions()
        LTA_attr = self.LTA_model.get_normalized_attributions()
        LR_attr = self.LR_model.get_normalized_attributions()
        LSTM_attr = self.LSTM_model.get_normalized_attributions()
        attributions = {'SP': SP_attr, 'LTA': LTA_attr, 'LR': LR_attr, 'LSTM': LSTM_attr}
        if output:
            return attributions
        self.attributions = attributions

    def load_non_norm_attributions(self):
        SP_non_norm = self.SP_model.get_non_normalized_attributions()
        LTA_non_norm = self.LTA_model.get_non_normalized_attributions()
        LR_non_norm = self.LR_model.get_coefs()
        LSTM_non_norm = self.LSTM_model.get_non_normalized_attributions()
        return {'SP': sum(SP_non_norm), 'LTA': sum(LTA_non_norm), 'LR': sum(LR_non_norm), 'LSTM': sum(LSTM_non_norm)}

    def plot_attributions(self, print_sum_attr=True, cv=False):
        channel_names = []
        for ch_idx in range(len(self.idx_to_ch)):
            channel_names.append(self.idx_to_ch[ch_idx])

        df_means = pd.DataFrame({'Channel': channel_names,
                           'LTA Attribution': self.attributions['LTA'],
                           'SP Attribution': self.attributions['SP'],
                           'LR Attribution': self.attributions['LR'],
                           'LSTM Attribution': self.attributions['LSTM']})
        if cv:
            df_std = pd.DataFrame({'LTA Attribution': self.attributions_std['LTA'],
                                   'SP Attribution': self.attributions_std['SP'],
                                   'LR Attribution': self.attributions_std['LR'],
                                   'LSTM Attribution': self.attributions_std['LSTM']})
            yerr = df_std.values.T
        else:
            yerr = 0

        ax = df_means.plot.bar(x='Channel', rot=90, yerr=yerr, capsize=3)
        if print_sum_attr and not cv:
            ax.legend(['LTA Attribution (sum ' + str(round(self.load_non_norm_attributions()['LTA'], 2)) + ')',
                       'SP Attribution (sum ' + str(round(self.load_non_norm_attributions()['SP'], 2)) + ')',
                       'LR Attribution (sum ' + str(round(self.load_non_norm_attributions()['LR'], 2)) + ')',
                       'LSTM Attribution (sum ' + str(round(self.load_non_norm_attributions()['LSTM'], 2)) + ')'])
        ax.set_xlabel("Source / Medium")
        plt.tight_layout()
        plt.title('Attributions', fontsize=16)
        plt.show()
        self.plot_touchpoint_attributions()

    def plot_touchpoint_attributions(self, max_seq_len=5):
        for seq_len in range(2, max_seq_len+1):
            touchpoint_attr = self.LSTM_model.get_touchpoint_attr(seq_len)
            plt.plot(touchpoint_attr, marker='.', linewidth=2, markersize=12)
            plt.title('Touchpoint attention attributions')
            plt.xlabel('Touchpoint index')
            plt.ylabel('Normalized attention')
        plt.show()

    def add_custom_attr(self):
        if self.custom_attr_eval is not None:
            attr = np.zeros(self.nr_top_ch)
            for ch_name in self.custom_attr_eval:
                idx = self.ch_to_idx[ch_name]
                attr[idx] = self.custom_attr_eval[ch_name]
            attr = attr / attr.sum()
            self.attributions['custom'] = attr.tolist()

    def profit_eval(self):
        if self.simulate:
            print('Oops... Can\'t run eval FW with simulated data')
            return
        GA_nonratio_df = self.data_loader.get_GA_nonratio_df()
        converted_clients_df = self.data_loader.get_converted_clients_df()

        eval_results_df = pd.DataFrame()
        for model_name in self.attributions:
            evaluation = Evaluation(GA_nonratio_df, converted_clients_df, self.total_budget, self.attributions[model_name], self.ch_to_idx)
            results = evaluation.evaluate()
            results['model'] = model_name
            eval_results_df = eval_results_df.append(results, ignore_index=True)
        print(eval_results_df)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-03-10.csv'
    start_date_data = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date_data = pd.Timestamp(year=2021, month=3, day=9, hour=23, minute=59, tz='UTC')

    start_date_cohort = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date_cohort = pd.Timestamp(year=2021, month=2, day=28, hour=23, minute=59, tz='UTC')

    train_proportion = 0.8
    nr_top_ch = 10
    ratio_maj_min_class = 1
    use_time = True
    total_budget = 5000

    simulate = False
    cohort_size = 10000
    sim_time = 100

    epochs = 10
    batch_size = 20
    learning_rate = 0.001

    ctrl_var = None
    ctrl_var_value = None
    eval_fw = True
    custom_attr_eval = {'google / cpc': 1,
                        'facebook / ad': 1,
                        'mecenat / partnership': 1,
                        'studentkortet / partnership': 1,
                        'tiktok / ad': 1,
                        'adtraction / affiliate': 1,
                        'snapchat / ad': 1}

    experiments = Experiments(start_date_data, end_date_data, start_date_cohort, end_date_cohort,
                              file_path_mp, nr_top_ch, train_proportion, ratio_maj_min_class, use_time,
                              simulate, cohort_size, sim_time, epochs, batch_size, learning_rate, ctrl_var,
                              ctrl_var_value, eval_fw, total_budget, custom_attr_eval)
    #experiments.cv(total_budget)

    experiments.init_models()
    experiments.load_data()
    experiments.train_all()
    experiments.load_attributions()
    experiments.validate_pred()
    experiments.plot_attributions(print_sum_attr=False)
