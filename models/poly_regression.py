from models.model import Model
import config
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


class PolyRegression(Model):

    def polynomial_regression(self, X, y, d):
        # sc_y = StandardScaler() #y scale
        # y = sc_y.fit_transform(y) #y scale

        data_set = self.reshape_to_df(X, pd.DataFrame(y))
        data_set.index = pd.DatetimeIndex(data_set.index)
        data_set.sort_index(inplace=True)  # data is monotonic
        X = pd.DataFrame(data_set.index)
        X.columns = ["Date"]
        # X['Date'] = X['Date'].map(dt.datetime.toordinal)
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

        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=d)),
            ('linreg', LinearRegression(normalize=True))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_train)
        rmse1, mape1 = self.calc_err_measures(y_train, y_pred)

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
        plt.plot(X_train, y_pred, color='orange', label='Model Fit')

        # plt.show()
        rmse1, mape1 = self.calc_err_measures(y_train, y_pred)

        # y_pred = pol_reg.predict(poly_reg.fit_transform(X_test))

        y_pred = pipeline.predict(X_test)
        # y_pred = sc_y.inverse_transform(y_pred) #y scale

        # y_test = y_test * 8
        # y_pred = y_pred * 8
        rmse, mape = self.calc_err_measures(y_test, y_pred)

        # plt.plot(X_test, y_test, color='blue', label='Data')

        plt.xticks(rotation=config.ROTATION)
        plt.plot(X_test, y_pred, color='red', label='Prediction')
        plt.xlabel('Date')
        plt.ylabel('Case')
        # plt.title(geo_id + ', Poly Regression')   # + '\nRMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
        plt.legend(loc='best')
        # plt.savefig(geo_id + '_Poly.png')
        plt.clf()
        # return mape1, mape
        return rmse, mape