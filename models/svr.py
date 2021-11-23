from models.model import Model
import config
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from mlxtend.evaluate import bias_variance_decomp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from pandas import concat
from numpy import asarray


class Svr(Model):
    def __init__(self, geo_id):
        self.geo_id = geo_id

    def series_to_supervised(self, df, n_in=1, n_out=1, dropnan=True):
        # n_vars = 1 if type(data) is list else data.shape[1]
        # df = pd.DataFrame(data)
        cols = list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # put it all together
        agg = concat(cols, axis=1)
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg.values

    def train_test_split(self, data, n_test):
        return data[:-n_test, :], data[-n_test:, :]

    def walk_forward_validation(self, data, n_test):
        predictions = list()
        # split dataset
        train, test = self.train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # split test row into input and output columns
            testX, testy = test[i, :-1], test[i, -1]
            # fit model on history and make a prediction
            yhat = self.svr_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop

            for k in range(0, 7 - i):
                test[i + k, -1 - 2 * k] = yhat
            history.append(test[i])
            # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
        # estimate prediction error
        error = mean_squared_error(test[:, -1], predictions)
        return error, test[:, 1], predictions
    # Tuning of parameters for regression by cross-validation

    def cross_valid(self, X, y):
        K = 15  # Number of cross valiations
        # Parameters for tuning
        parameters = [
            {'kernel': ['rbf'],
             'gamma': ['scale'],
             # 'epsilon': [0.0001, 0.001, 0.01, 0.1, 0.25],
             # 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.7, 0.9, 'scale'],
             # 'C': [100, 285, 500, 1000, 5000, 10000]}
             'C': [10, 100, 1000, 2000, 5000, 10000, 100000]}
        ]
        # print("Tuning hyper-parameters")
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        svr = GridSearchCV(SVR(), parameters, cv=K, scoring=scorer)
        svr.fit(X, y.values.ravel())

        # Checking the score for all parameters
        means = svr.cv_results_['mean_test_score']
        stds = svr.cv_results_['std_test_score']

        min_val = max(means)
        min_index = list(means).index(min_val)
        print(parameters[0]['C'][min_index])
        # for mean, std, params in zip(means, stds, svr.cv_results_['params']):
        #     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    def svr_forecast(self, train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
        # fit model
        model = SVR(kernel='rbf') #C=config.SVR_C, gamma=config.SVR_GAMMA
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict([testX])
        return yhat[0]

    def svr_diffed(self, df, geo_id):
        df['Date'] = df.index
        data = self.series_to_supervised(df, n_in=7)

        mse, y, yhat = self.walk_forward_validation(data, 7)
        print('mse', mse)
        plt.plot(y, label='Expected')
        plt.plot(yhat, label='Predicted')
        plt.legend()
        plt.show()
        return

    def svr(self, X, y):
        X_train = X[:-1 * config.END_DATE]
        y_train = y[:-1 * config.END_DATE]
        X_test = X[-1 * config.END_DATE: -1 * config.START_DATE]
        y_test = y[-1 * config.END_DATE: -1 * config.START_DATE]
        # print(len(X))

        regression = SVR(kernel='rbf', C=config.SVR_C, gamma=config.SVR_GAMMA)
        # regression = SVR(kernel='linear', max_iter=10000)
        # regression = SVR(kernel='poly', degree=7)
        # regression = SVR(kernel='sigmoid', gamma=0.0001)

        # self.print_bias_variance(regression, X_train.values, y_train.values.ravel(),
        #                          X_test.values, y_test.values.ravel(), geo_id)

        # y_train.values.reshape(-1, 1)
        regression.fit(X_train, y_train.values.ravel())
        predict_date = pd.DataFrame(X_test)
        y_predict = regression.predict(predict_date)  # predict test data

        ly_test = [data[0] for data in y_test.values]
        y_test = np.array(ly_test)
        ly_train = [data[0] for data in y_train.values]
        y_train = np.array(ly_train)

        # plt.xticks(rotation=config.ROTATION)
        # plt.xlabel('Date')
        # plt.ylabel('Case')
        # plt.plot(X, y, color='blue', label='Data')
        # plt.plot(X_test, y_predict, color='red', label='Prediction')

        rmse, mape = self.calc_err_measures(y_test, y_predict)
        # plt.title(geo_id + ', SVR')    # + '\n RMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
        # plt.legend(loc='best')
        # plt.show()

        predict_date = pd.DataFrame(X_train)
        y_predictt = regression.predict(predict_date)  # predict train data
        rmse1, mape1 = self.calc_err_measures(y_train, y_predictt)

        # plt.title(f'{geo_id}, SVR')
        # plt.plot(X_train, y_predict, color='orange', label='Model Fit')
        # plt.legend(loc='best')
        # plt.show()
        # plt.savefig(f'SVR_{geo_id}.png')
        # plt.clf()
        return rmse, mape, y_predict, y_predictt
