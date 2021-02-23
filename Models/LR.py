import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt
from ModelDataLoader import ModelDataLoader

class LR:
    def __init__(self):
        self.log_reg = LogisticRegression()
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_train_test_data(self, x_train, y_train, x_test, y_test):
        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

    def train(self):
        self.log_reg.fit(self.x_train, self.y_train)

    def get_coefs(self):
        return self.log_reg.coef_[0]

    def get_predictions(self, x):
        return self.log_reg.predict(x)

    def get_results(self):
        preds = self.get_predictions(self.x_test)
        tn, fp, fn, tp = confusion_matrix(self.y_test, preds).ravel()
        auc = roc_auc_score(self.y_test, preds)
        logloss = log_loss(self.y_test, preds)

        return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'auc': auc, 'logloss': logloss}

    def validate(self):
        res = self.get_results()
        print('Accuracy ', (res['tn'] + res['tp']) / (res['tn'] + res['tp'] + res['fp'] + res['fn']))
        print('AUC: ', res['auc'])
        print('Log-loss: ', res['logloss'])
        print('tn:', res['tn'], ' fp:', res['fp'], ' fn:', res['fn'], ' tp:', res['tp'])
        print('precision: ', res['tp'] / (res['tp'] + res['fp']))
        print('recall: ', res['tp'] / (res['tp'] + res['fn']))

    def plot_attributions(self, idx_to_ch):
        coefs = self.get_coefs()
        ch_names = []
        for idx, _ in enumerate(coefs):
            ch_names.append(idx_to_ch[idx])
        df = pd.DataFrame({'Channel': ch_names, 'Attribution': coefs})
        ax = df.plot.bar(x='Channel', y='Attribution', rot=90)
        plt.tight_layout()
        plt.title('Attributions - LR model')
        plt.axhline([0])
        plt.show()

    def get_normalized_attributions(self):
        coefs = self.get_coefs()
        coefs[coefs < 0] = 0
        channel_attributions = coefs/coefs.sum()
        return channel_attributions.tolist()

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-22.csv'
    start_date_data = pd.Timestamp(year=2021, month=2, day=2, hour=0, minute=0, tz='UTC')
    end_date_data = pd.Timestamp(year=2021, month=2, day=21, hour=23, minute=59, tz='UTC')

    start_date_cohort = pd.Timestamp(year=2021, month=2, day=5, hour=0, minute=0, tz='UTC')
    end_date_cohort = pd.Timestamp(year=2021, month=2, day=13, hour=23, minute=59, tz='UTC')

    train_proportion = 0.8
    nr_top_ch = 10
    ratio_maj_min_class = 1
    use_time = True

    data_loader = ModelDataLoader(start_date_data, end_date_data, start_date_cohort, end_date_cohort, file_path_mp,
                                  nr_top_ch, ratio_maj_min_class)
    x_train, y_train, x_test, y_test = data_loader.get_feature_matrix_split(train_proportion, use_time)

    model = LR()
    model.load_train_test_data(x_train, y_train, x_test, y_test)
    model.train()
    model.validate()
    model.plot_attributions(data_loader.get_idx_to_ch_map())