import config
import pandas as pd
import numpy as np
import math
from mlxtend.evaluate import bias_variance_decomp

mse1 = 0
bias1 = 0
var1 = 0
count1 = 0


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

    def print_bias_variance(self, model, X_train, y_train, X_test, y_test, geo_id):
        mse, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test,
                                              loss='mse', num_rounds=200, random_seed=1)

        print("Bias Variance Calculations")
        print('MSE : %.3f' % mse)
        print('Bias: %.3f' % bias)
        print('Var : %.3f' % var)

        f = open('/Trend/RF_EBV.csv', 'a')
        f.write(geo_id + f',{"{:.2f}".format(mse)},{"{:.2f}".format(bias)},{"{:.2f}".format(var)}\n')
        f.close()

        global mse1, bias1, var1, count1
        mse1 += mse
        bias1 += bias
        var1 += var
        count1 += 1
