from models.model import Model
import config
import numpy as np
import pandas as pd
from pandas import concat
from sklearn import ensemble as ml
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
# from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot


class RandomForest(Model):

    def __init__(self, geo_id):
        self.geo_id = geo_id

    def cross_valid(self, X, y):
        K = 15  # Number of cross valiations
        # Parameters for tuning
        parameters = [
            {
             'criterion': ['squared_error'],
             'random_state': [1, 10, 100],
             # 'n_estimators': [1, 10, 100],
             'n_estimators': [50],
                'max_depth': [3, 5, 10, 50],

            }
        ]
        # print("Tuning hyper-parameters")
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        rf = GridSearchCV(ml.RandomForestRegressor(bootstrap=False), parameters, cv=K, scoring=scorer)
        rf.fit(X, y.values.ravel())

        # Checking the score for all parameters
        means = rf.cv_results_['mean_test_score']
        stds = rf.cv_results_['std_test_score']

        # min_val = max(means)
        # min_index = list(means).index(min_val)
        # print(parameters[0]['C'][min_index])
        for mean, std, params in zip(means, stds, rf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

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

    def walk_forward_test(self, data, n_test):
        predictions = list()
        # split dataset
        train, test = self.train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        # print(test)
        for i in range(len(test)):
            # split test row into input and output columns
            # testX, testy = test[i, :-1], test[i, -1]
            testX, testy = test[i, :-1], test[i, -1]

            # fit model on history and make a prediction
            yhat = self.random_forest_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop

            for k in range(0, 14 - i):
                test[i + k, -1 - 2 * k] = yhat
            history.append(test[i])
            # summarize progress
            # print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
        # estimate prediction error
        error = mean_squared_error(test[:, -1], predictions)
        return error, test[:, 1], predictions

    def walk_forward_validation(self, data, n_test):
        predictions = list()
        # split dataset
        train, test = self.train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        # print(test)
        for i in range(len(test)):
            # split test row into input and output columns
            # testX, testy = test[i, :-1], test[i, -1]
            testX, testy = test[i, :-1], test[i, -1]

            # fit model on history and make a prediction
            yhat = self.random_forest_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
            # summarize progress
            # print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
        # estimate prediction error
        error = mean_squared_error(test[:, -1], predictions)
        return error, test[:, 1], predictions

    def random_forest_forecast(self, train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
        # print('training on')
        # print(trainy)
        # fit model
        model = RandomForestRegressor(n_estimators=100)
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict([testX])
        return yhat[0]

    def rf_reg_diffed(self, df):
        df['Date'] = df.index
        data = self.series_to_supervised(df, n_in=14)
        # print('dude')
        # print(df)
        X = df['Date']
        X_train = X[:-1 * config.END_DATE]
        X_test = X[-1 * config.END_DATE: -1 * config.START_DATE]
        # y = df['Cases']
        mse, y0, ypred = self.walk_forward_test(data, 14)
        mse1, y1, ytrain = self.walk_forward_validation(data, len(data) - 15)
        # print('mse', mse)
        # pyplot.plot(y, label='Expected')
        # pyplot.plot(yhat, label='Predicted')
        # pyplot.legend()
        # pyplot.show()

        # plt.xticks(rotation=config.ROTATION)
        # plt.title(f'{geo_id}, RF Regression')   # + '\n RMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
        # plt.plot(X, y, color='blue', label='Data')
        # plt.plot(X_train[14:], ytrain, color='orange', label='Model Fit')
        # plt.plot(X_test, ypred, color='red', label='Prediction')
        # plt.xlabel('Date')
        # plt.ylabel('Case')
        # plt.legend(loc='best')
        # plt.savefig(f'RF_{geo_id}.png')
        # plt.clf()
        # plt.show()
        rmse, mape = self.calc_err_measures(np.array(y0), ypred)
        # print(rmse, mape)
        return rmse, mape, ypred, ytrain

    def rf_regression(self, X, y):
        regression = ml.RandomForestRegressor(n_estimators=1, max_depth=15,
                                              min_samples_split=2, criterion='squared_error')

        X_train = X[:-1 * config.END_DATE]
        y_train = y[:-1 * config.END_DATE]
        X_test = X[-1 * config.END_DATE: -1 * config.START_DATE]
        y_test = y[-1 * config.END_DATE: -1 * config.START_DATE]

        ly_train = [data[0] for data in y_train.values]
        ly_test = [data[0] for data in y_test.values]
        y_train = ly_train
        y_test = ly_test

        regression.fit(X_train.values, y_train[1:])
        y_predictt = regression.predict(X_train.values)
        rmse_train, mape_train = self.calc_err_measures(np.array(y_train), y_predictt)
        print('train data rmse and mape', rmse_train, mape_train)
        plt.plot(X, y, color='blue', label='Data')
        plt.plot(X_train, y_predictt, color='orange', label='Model Fit')
        y_predict = regression.predict(X_test.values)
        rmse, mape = self.calc_err_measures(np.array(y_test), y_predict)
        # print('test_data', rmse, mape)
        # print("=================================")
        # plt.plot(X_test, y_test, color='blue', label=' Data')
        # plt.plot(X_test, y_predict, color='red', label='Prediction')
        # plt.xticks(rotation=config.ROTATION)
        # plt.title(f'{geo_id}, RF Regression')   # + '\n RMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
        # plt.plot(X, y, color='blue', label='Data')
        # plt.plot(X_train, y_predict, color='orange', label='Model Fit')
        # plt.plot(X_test, y_predict, color='red', label='Prediction')
        # plt.xlabel('Date')
        # plt.ylabel('Case')
        # plt.legend(loc='best')
        # plt.show()
        # plt.savefig(geo_id + '_RF.png')
        # plt.clf()
        return rmse, mape, y_predict, y_predictt
