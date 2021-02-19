import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import matplotlib.pyplot as plt

class LR:
    def __init__(self):
        self.log_reg = LogisticRegression()
        #self.idx_to_ch = self.data_loader.get_idx_to_ch_map()
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

    def plot_attributions(self):
        coefs = self.get_coefs()
        ch_names = []
        for idx, _ in enumerate(coefs):
            ch_names.append(self.idx_to_ch[idx])
        df = pd.DataFrame({'Channel': ch_names, 'Attribution': coefs})
        ax = df.plot.bar(x='Channel', y='Attribution', rot=90)
        plt.tight_layout()
        plt.title('Attributions - LR model')
        plt.axhline([0])
        plt.show()

    def get_normalized_attributions(self):
        coefs = self.get_coefs()
        minimum = coefs.min()
        if minimum < 0:
            coefs += abs(minimum)
        channel_attributions = coefs/coefs.sum()
        return channel_attributions.tolist()

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    file_path_mp = '../Data/Mixpanel_data_2021-02-17.csv'
    start_date = pd.Timestamp(year=2021, month=2, day=3, hour=0, minute=0, tz='UTC')
    end_date = pd.Timestamp(year=2021, month=2, day=10, hour=23, minute=59, tz='UTC')

    train_proportion = 0.7
    nr_top_ch = 10
    ratio_maj_min_class = 1

    model = LR()
    model.train()
    model.validate()
    model.plot_attributions()