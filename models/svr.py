from models.model import Model
import config
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVR


class Svr(Model):
    def svr(self, X, y, gamma, c):
        # sc_y = StandardScaler()
        # y = sc_y.fit_transform(y)

        # n = 12
        X_train = X[:-1 * config.END_DATE]
        y_train = y[:-1 * config.END_DATE]
        X_test = X[-1 * config.END_DATE: -1 * config.START_DATE]
        y_test = y[-1 * config.END_DATE: -1 * config.START_DATE]

        regression = SVR(kernel='rbf', C=285, gamma='scale')
        # regression = SVR(kernel='linear', max_iter=10000)
        # regression = SVR(kernel='poly', degree=7)
        # regression = SVR(kernel='sigmoid', gamma='auto')
        regression.fit(X_train, y_train)
        predict_date = pd.DataFrame(X_test)
        y_pred = regression.predict(predict_date)

        # print(y_train)
        # y_pred = list(y_pred)

        ly_test = [data[0] for data in y_test.values]
        y_test = np.array(ly_test)

        ly_train = [data[0] for data in y_train.values]
        y_train = np.array(ly_train)

        # y_pred = sc_y.inverse_transform(y_pred)
        # y_test = sc_y.inverse_transform(y_test)

        plt.xticks(rotation=config.ROTATION)
        plt.plot(X, y, color='blue', label='Data')
        plt.plot(X_test, y_pred, color='red', label='Prediction')
        # plt.plot(X_test, y_test, color='blue', label='Predicted Data')

        rmse, mape = self.calc_err_measures(y_test, y_pred)
        plt.xlabel('Date')
        plt.ylabel('Case')
        # plt.title(geo_id + ', SVR')    # + '\n RMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
        plt.legend(loc='best')
        # plt.show()

        predict_date = pd.DataFrame(X_train)
        y_pred = regression.predict(predict_date)
        # y_pred = sc_y.inverse_transform(y_pred)
        # y_train = sc_y.inverse_transform(y_train)
        # y = sc_y.inverse_transform(y)

        plt.plot(X_train, y_pred, color='orange', label='Model Fit')
        # plt.plot(X_train, y_train, color='blue')

        # print(y_test)
        # print(y_pred)
        rmse1, mape1 = self.calc_err_measures(y_train, y_pred)
        # plt.show()
        plt.legend(loc='best')
        # plt.savefig(geo_id + '_SVR.png')
        plt.clf()
        return rmse, mape
