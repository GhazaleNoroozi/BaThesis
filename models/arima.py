from models.model import Model
import config
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


class Arima(Model):

    def check_trend(self, data_set):
        decomposition = seasonal_decompose(data_set)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(np.log(data_set), label='Original', color='blue')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(trend, label='Trend', color='blue')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonality', color='blue')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residuals', color='blue')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        return

    def check_stationary(self, data_set):
        data_set = data_set.diff(periods=6).dropna()
        data_set = data_set.diff(periods=1).dropna()
        # data_set = data_set.diff(periods=1).dropna()
        rol_mean = data_set.rolling(window=8, center=False).mean()
        rol_std = data_set.rolling(window=8, center=False).std()

        fig = plt.figure(figsize=(12, 6))
        orig = plt.plot(data_set, color='blue', label='Original')
        mean = plt.plot(rol_mean, color='purple', label='Rolling Mean')
        std = plt.plot(rol_std, color='pink', label='Rolling Std')
        plt.legend(loc='best')
        plt.show()
        # print("Results for the Dickey-Fuller test")
        dftest = adfuller(data_set['Case'])
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        # print(dfoutput[1])

        self.plot_acf_pacf(data_set)

        return dfoutput[1]

    def plot_acf_pacf(self, data_set):
        fig, ax = plt.subplots(figsize=(8, 3))
        plot_acf(data_set, ax=ax, lags=10)
        plt.show()
        fig, ax = plt.subplots(figsize=(8, 3))
        plot_pacf(data_set, ax=ax, lags=10)
        plt.show()

    def arima(self, X, y, order):
        data_set = self.reshape_to_df(X, pd.DataFrame(y))
        data_set.index = pd.DatetimeIndex(data_set.index)
        data_set.sort_index(inplace=True)  # data is monotonic

        print_me = self.check_stationary(data_set)
        # plot_acf_pacf(data_set)

        train_data = data_set.copy(deep=True)
        train_data = train_data.loc[:pd.to_datetime(config.LAST_DATE, dayfirst=True)
        - datetime.timedelta(days=config.END_DATE - 1), :]
        model = ARIMA(train_data['Case'], order=order)
        model_fit = model.fit()
        # print(model_fit.model.predict())
        # print(model_fit.summary())

        fitted = model_fit.predict(
            start=pd.to_datetime('02/03/2021', dayfirst=True),
            end=pd.to_datetime(config.LAST_DATE, dayfirst=True)
            - datetime.timedelta(days=config.END_DATE), dynamic=False)

        predicted = model_fit.predict(
            start=pd.to_datetime(config.LAST_DATE, dayfirst=True)
            - datetime.timedelta(days=config.END_DATE - 1),
            end=pd.to_datetime(config.LAST_DATE, dayfirst=True)
            - datetime.timedelta(days=config.START_DATE), dynamic=True)

        data_set['Predicted'] = predicted
        data_set['Fitted'] = fitted

        y_test = data_set['Case'][-1 * config.END_DATE: -1 * config.START_DATE]
        y_pred = data_set['Predicted'].dropna()
        y_fitted = data_set['Fitted'].dropna()
        # X = train_data.index
        X_test = X[-1 * config.END_DATE: -1 * config.START_DATE]
        X_train = X[:-1 * config.END_DATE]
        rmse, mape = self.calc_err_measures(y_test, y_pred)
        # plt.plot(X_test, y_test, color='blue', label='Data')

        # plt.xticks(rotation=rotation)
        # plt.plot(X, data_set['Case'], color='blue', label='Data')
        # plt.plot(X_train, y_fitted, color='orange', label='Model Fit')
        # plt.plot(X_test, y_pred, color='red', label='Prediction')
        # plt.xlabel('Date')
        # plt.ylabel('Case')
        # plt.title(geo_id + ', ARIMA')   # + '\n RMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
        # plt.legend(loc='best')
        # plt.save fig(geo_id + '_ARIMA.png')
        # plt.clf()
        return rmse, mape, print_me
