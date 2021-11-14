import config
import pandas as pd
import numpy as np
import math


class Model:
    def reshape_to_df(self, X, y):
        ly = []
        for i in y.values:
            ly.append(i[0])
        lx = []
        for i in X.values:
            lx.append(i[0])

        data_set = pd.DataFrame({'Date': lx, 'Case': ly})
        # data_set['Case1'] = data_set['Case'] - data_set['Case'].shift(1)
        # data_set['Case2'] = data_set['Case'] - data_set['Case'].shift(2)

        # print(data_set)
        # fig = plt.figure(figsize=(12, 8))
        data_set.set_index('Date', inplace=True)
        # fig.add_subplot(211)
        # test_result = adfuller(data_set['Case'])
        # test_result = adfuller(data_set['Case1'].dropna())
        # test_result = adfuller(data_set['Case2'].dropna())

        # fig = data_set['Case1'].plot()
        # ax2 = fig.add_subplot(212)
        # fig = data_set['Case2'].plot()
        # ax3 = fig.add_subplot(221)
        # fig = sm.graphics.tsa.plot_pacf(data_set['Seasonal First Difference'].dropna(), lags=40, ax=ax2)

        # data_set.plot(subplots=True)
        # data_set['Case1'].plot()
        # data_set['Case2'].plot()
        # plt.show()

        return data_set

    def calc_err_measures(self, y_test, y_pred):
        diff = []
        for i in range(len(y_test)):
            if y_test[i] > 10 if config.CASE_OR_DEATH == 'c' else 0:
                diff.append((y_test[i] - y_pred[i]) / y_test[i])

        mape = int(np.mean(np.abs(diff) / len(diff) * 10000))
        mape = mape / 100

        rmse = math.sqrt(np.mean((y_test - y_pred) * (y_test - y_pred)))
        # rmse = rmse / (np.max(y_test) - np.min(y_test))
        rmse = rmse / len(y_test)
        rmse = int(rmse * 10000)
        rmse = rmse / 100
        return rmse, mape