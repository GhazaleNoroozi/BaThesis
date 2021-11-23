from models.model import Model
import config
import pandas as pd
import numpy as np
from pandas import DataFrame
import datetime as dt
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import mean_squared_error
from pandas import concat
from numpy import asarray


class PolyRegression(Model):
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
            yhat = self.pr_forecast(history, testX)
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop

            for k in range(0, 14 - i):
                test[i + k, -1 - 2 * k] = yhat
            history.append(test[i])
            # summarize progress
            print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
        # estimate prediction error
        error = mean_squared_error(test[:, -1], predictions)
        return error, test[:, 1], predictions

    def pr_forecast(self, train, testX):
        # transform list into array
        train = asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
        # fit model
        scaler = StandardScaler()
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=config.PR_DIMENSION)),
            ('scaler', scaler),
            ('linreg', LinearRegression())
            # ('linreg', LogisticRegression(solver='newton-cg'))
        ])
        model.fit(trainX, trainy)
        # make a one-step prediction
        yhat = model.predict([testX])
        return yhat[0]

    def pr_diffed(self, df):
        df['Date'] = df.index
        data = self.series_to_supervised(df, n_in=14)

        mse, y, yhat = self.walk_forward_validation(data, 14)
        print('mse', mse)
        plt.plot(y, label='Expected')
        plt.plot(yhat, label='Predicted')
        plt.legend()
        plt.show()
        return

    def polynomial_regression(self, X, y):
        # sc_y = StandardScaler() #y scale
        # y = sc_y.fit_transform(y) #y scale

        data_set = self.reshape_to_df(X, pd.DataFrame(y))
        data_set.index = pd.DatetimeIndex(data_set.index)
        data_set.sort_index(inplace=True)  # data is monotonic
        X = pd.DataFrame(data_set.index)
        X.columns = ["Date"]
        X['Date'] = X['Date'].map(dt.datetime.toordinal)
        X = X.iloc[:, 0:1].values
        x = []
        for i in range(0, len(X)):
            x.append([i])
        # X = x
        y = data_set.iloc[:, 0].values
        # y = y/8

        X_train = X[:-1 * config.END_DATE]
        y_train = y[:-1 * config.END_DATE]
        X_test = X[-1 * config.END_DATE: -1 * config.START_DATE]
        y_test = y[-1 * config.END_DATE: -1 * config.START_DATE]
        scalar = StandardScaler()
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=config.PR_DIMENSION)),
            ('scaler', scalar),
            ('linreg', LinearRegression())
            # ('linreg', LogisticRegression(solver='newton-cg'))
        ])

        pipeline.fit(X_train, y_train)
        y_pred_train = pipeline.predict(X_train)
        rmse1, mape1 = self.calc_err_measures(y_train, y_pred_train)

        # poly_reg = PolynomialFeatures(degree=d)
        # X_poly = poly_reg.fit_transform(X_train)
        # pol_reg = LinearRegression(normalize=False)
        # pol_reg.fit(X_poly, y_train)
        # y_pred = pol_reg.predict(poly_reg.fit_transform(X_train))

        # y_test = sc_y.inverse_transform(y_test) #y scale
        # y_pred = sc_y.inverse_transform(y_pred) #y scale
        # y_train = sc_y.inverse_transform(y_train) #y scale

        # plt.plot(X_train, y_train, color='blue')
        plt.plot(X, y, color='blue', label='Data')
        plt.plot(X_train, y_pred_train, color='orange', label='Model Fit')

        # plt.show()
        rmse1, mape1 = self.calc_err_measures(y_train, y_pred_train)

        # y_pred = pol_reg.predict(poly_reg.fit_transform(X_test))

        y_pred_test = pipeline.predict(X_test)
        # y_pred = sc_y.inverse_transform(y_pred) #y scale

        # y_test = y_test * 8
        # y_pred = y_pred * 8
        rmse, mape = self.calc_err_measures(y_test, y_pred_test)

        # self.print_bias_variance(pipeline, X_train, y_train, X_test, y_test)

        # plt.plot(X_test, y_test, color='blue', label='Data')

        # plt.xticks(rotation=config.ROTATION)
        # plt.plot(X_test, y_pred, color='red', label='Prediction')
        # plt.xlabel('Date')
        # plt.ylabel('Case')
        # plt.title(f'{self.geo_id}, Poly Regression')   # + '\nRMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
        # plt.legend(loc='best')
        # plt.show()

        # plt.savefig(f'Poly_{self.geo_id}.png')
        # plt.clf()
        # return mape1, mape
        return rmse, mape,  y_pred_test, y_pred_train
