from models.model import Model
import config
import numpy as np
import pandas as pd
from sklearn import ensemble as ml
from matplotlib import pyplot as plt


class RandomForest(Model):

    def rf_regression(self, X, y):
        # sc_y = StandardScaler()
        # y = sc_y.fit_transform(y)

        regressor = ml.RandomForestRegressor(n_estimators=100, random_state=1, criterion='mse')

        # n = 12
        X_train = X[:-1 * config.END_DATE]
        y_train = y[:-1 * config.END_DATE]
        X_test = X[-1 * config.END_DATE: -1 * config.START_DATE]
        y_test = y[-1 * config.END_DATE: -1 * config.START_DATE]

        ly_train = [data[0] for data in y_train.values]
        ly_test = [data[0] for data in y_test.values]
        y_train = ly_train
        y_test = ly_test
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(pd.DataFrame(X_train))
        rmse1, mape1 = self.calc_err_measures(np.array(y_train), y_pred)

        # y_test = sc_y.inverse_transform(y_test)
        # y_pred = sc_y.inverse_transform(y_pred)
        # y_train = sc_y.inverse_transform(y_train)

        # plt.plot(X_train, y_train, color='blue') #now
        plt.plot(X, y, color='blue', label='Data')
        plt.plot(X_train, y_pred, color='orange', label='Model Fit')
        # plt.show()

        y_pred = regressor.predict(pd.DataFrame(X_test))
        # y_pred = sc_y.inverse_transform(y_pred)
        rmse, mape = self.calc_err_measures(np.array(y_test), y_pred)

        plt.xticks(rotation=config.ROTATION)
        # plt.plot(X_test, y_test, color='blue', label=' Data')

        plt.plot(X_test, y_pred, color='red', label='Prediction')
        plt.xlabel('Date')
        plt.ylabel('Case')
        # plt.title(geo_id + ', RF Regression')   # + '\n RMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
        plt.legend(loc='best')
        # plt.savefig(geo_id + '_RF.png')
        plt.clf()
        return rmse, mape
        # return mape1, mape