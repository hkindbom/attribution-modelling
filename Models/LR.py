from DataProcessing import DataProcessing
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
from ModelDataLoader import ModelDataLoader

class LR:
    def __init__(self, start_date, end_date, file_path_mp, nr_top_ch, train_prop=0.8, ratio_maj_min_class=1):
        self.data_loader = ModelDataLoader(start_date, end_date, file_path_mp, nr_top_ch, ratio_maj_min_class)
        self.log_reg = LogisticRegression()
        self.train_prop = train_prop
        self.clients_data_train = {}
        self.clients_data_test = {}
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_train_test_data(self):
        self.x_train, self.y_train, self.x_test, self.y_test = self.data_loader.get_feature_matrix_split(self.train_prop)

    def train(self):
        self.log_reg.fit(self.x_train, self.y_train)

    def get_coefs(self):
        return self.log_reg.coef_

    def get_predictions(self, x):
        return self.log_reg.predict(x)

    def validate(self):
        preds = self.get_predictions(self.x_test)
        auc = roc_auc_score(self.y_test, preds)
        logloss = log_loss(self.y_test, preds)
        tn, fp, fn, tp = confusion_matrix(self.y_test, preds).ravel()

        print('Accuracy ', (tn + tp) / (tn + tp + fp + fn))
        print('AUC: ', auc)
        print('Log-loss: ', logloss)
        print('tn:', tn, ' fp:', fp, ' fn:', fn, ' tp:', tp)
        print('precision: ', tp / (tp + fp))
        print('recall: ', tp / (tp + fn))

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-11.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=10, hour=23, minute=59, tz='UTC')

    train_proportion = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 1

    LR_model = LR(start_date, end_date, file_path_mp, nr_top_ch, train_proportion, ratio_maj_min_class)
    LR_model.load_train_test_data()
    LR_model.train()
    print(LR_model.get_coefs())
    LR_model.validate()