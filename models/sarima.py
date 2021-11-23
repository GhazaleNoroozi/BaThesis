from statsmodels.tsa.statespace import sarimax
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import datetime
import config
import datetime as dt
from models.model import Model
from mlxtend.evaluate import bias_variance_decomp
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


class Sarima(Model):

    def __init__(self, geo_id):
        self.geo_id = geo_id

    def power_spectrum(self, data):
        sampling_rate = 7
        fourier_transform = np.fft.rfft(data)

        abs_fourier_transform = np.abs(fourier_transform)

        power_spectrum = np.square(abs_fourier_transform)
        frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))

        # plt.plot(frequency, power_spectrum)
        # plt.show()
        return frequency

    def check_trend(self, data_set):
        decomposition = seasonal_decompose(data_set)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.title(self.geo_id)
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
        # plt.savefig(self.geo_id + '.png')
        plt.show()
        return

    def check_stationary(self, data_set):
        f = self.power_spectrum()

        # data_set = data_set.diff(periods=6).dropna()
        data_set = data_set.diff(periods=1).dropna()
        # data_set = data_set.diff(periods=1).dropna()
        rol_mean = data_set.rolling(window=14, center=False).mean()
        rol_std = data_set.rolling(window=14, center=False).std()

        # print("Results for the Dickey-Fuller test")
        dftest = adfuller(data_set['Case'])
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags used',
                                                 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

        fig = plt.figure(figsize=(12, 6))
        # plt.title(f'{self.geo_id}   p-value: %.2f' % dfoutput[1])
        plt.title(self.geo_id)
        orig = plt.plot(data_set, color='blue', label='Original')
        mean = plt.plot(rol_mean, color='purple', label='Rolling Mean')
        std = plt.plot(rol_std, color='pink', label='Rolling Std')
        plt.legend(loc='best')
        # plt.savefig(self.geo_id + '.png')
        plt.show()

        # self.plot_acf_pacf(data_set)
        return dfoutput[1]

    def plot_acf_pacf(self, data_set):
        fig, ax = plt.subplots(figsize=(8, 3))
        plot_acf(data_set, ax=ax, lags=14, title=self.geo_id + ' ACF')
        # plt.savefig(self.geo_id + '_acf.png')
        # plt.clf()
        plt.show()
        fig, ax = plt.subplots(figsize=(8, 3))
        plot_pacf(data_set, ax=ax, lags=14, title=self.geo_id + ' PACF')
        # plt.savefig(self.geo_id + '_pacf.png')
        # plt.clf()
        plt.show()

    def sarima(self, X, y):
        data_set = self.reshape_to_df(X, pd.DataFrame(y))
        data_set.index = pd.DatetimeIndex(data_set.index, freq=data_set.index.inferred_freq)
        data_set.sort_index(inplace=True)  # data is monotonic

        # print(f'power: {self.power_spectrum(data_set)}')
        # print(data_set.index.inferred_freq)
        # return
        # data_set = data_set.diff(periods=2).dropna()
        # # self.check_trend(data_set)
        # pvaleu =self.check_stationary(data_set)
        # print(f'pvalue: {pvaleu}')
        # return

        # X['Date'] = X['Date'].map(dt.datetime.toordinal)
        # self.power_spectrum(data_set)

        train_data = data_set.copy(deep=True)
        train_data = train_data.loc[:pd.to_datetime(config.LAST_DATE, dayfirst=True) -
                                    datetime.timedelta(days=config.END_DATE - 1), :]

        model = sarimax.SARIMAX(train_data, order=config.SARIMA_TREND, seasonal_order=config.SARIMA_SEASONAL)
        model_fit = model.fit(disp=0)

        fitted = model_fit.predict(start=pd.to_datetime('02/03/2021', dayfirst=True),
                                   end=pd.to_datetime(config.LAST_DATE, dayfirst=True)
                                   - datetime.timedelta(days=config.END_DATE), dynamic=False)
        data_set['Fitted'] = fitted

        predicted = model_fit.predict(start=pd.to_datetime(config.LAST_DATE, dayfirst=True)
                                      - datetime.timedelta(days=config.END_DATE - 1),
                                      end=pd.to_datetime(config.LAST_DATE, dayfirst=True)
                                      - datetime.timedelta(days=config.START_DATE),
                                      dynamic=True)

        data_set['Predicted'] = predicted

        y_test = data_set['Case'][-1 * config.END_DATE: -1 * config.START_DATE]
        y_train = data_set['Case'][:-1 * config.END_DATE]
        y_pred = data_set['Predicted'].dropna()
        y_fitted = data_set['Fitted'].dropna()
        # X = train_data.index

        X_test = X[-1 * config.END_DATE: -1 * config.START_DATE]
        X_train = X[:-1 * config.END_DATE]
        rmse, mape = self.calc_err_measures(y_test, y_pred)

        # self.print_bias_variance(model, np.array(X_train.values), np.array(y_train.values),
        #                          np.array(X_test.values), np.array(y_test.values), self.geo_id)
        #                          # loss='mse', num_rounds=200,random_seed=1)

        # plt.xticks(rotation=config.ROTATION)
        # plt.plot(X, data_set['Case'], color='blue', label='Data')
        # plt.plot(X_train, y_fitted, color='orange', label='Model Fit')
        # plt.plot(X_test, y_pred, color='red', label='Prediction')
        # plt.xlabel('Date')
        # plt.ylabel('Case')
        # plt.title(f'{self.geo_id}, SARIMA')   # + '\n RMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
        # plt.legend(loc='best')
        # plt.show()
        # plt.savefig(f'SARIMA_{self.geo_id}.png')
        # plt.clf()

        return rmse, mape, y_pred, y_fitted
