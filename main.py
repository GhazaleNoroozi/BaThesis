import pandas as pd
import config
from models.arima import Arima
from models.poly_regression import PolyRegression
from models.rf_regression import RandomForest
from models.svr import Svr


def read(geo_id):
    # dates are set as datetime format as index to the df
    df = pd.read_csv(config.file_name, index_col=0, parse_dates=True, dayfirst=True)
    # get data for the specified country with id = geoId
    geo_df = df.loc[df.geoId == geo_id]
    # make the data index monotonic
    df.sort_index(inplace=True)
    return geo_df


def create_features(df):
    geo_df = df.iloc[:, :6]
    X: pd.core.frame.DataFrame = pd.DataFrame(geo_df.index)
    y: pd.core.series.Series = geo_df.iloc[:, 3 if config.CASE_OR_DEATH == 'c' else 4]
    y: pd.core.frame.DataFrame = pd.DataFrame(y)
    return X, y


def visualize_data(df):
    pass


def models(X, y):
    if config.MODEL['arima'] == 1:
        Arima().arima(X, y, config.ARIMA_ORDER)
    if config.MODEL['poly reg'] == 1:
        PolyRegression().polynomial_regression(X, y, config.PR_DIMENSION)
    if config.MODEL['rf reg'] == 1:
        RandomForest().rf_regression(X, y)
    if config.MODEL['svr'] == 1:
        Svr().svr(X, y, config.SVR_GAMMA, config.SVR_C)


def main():
    for geo_id in config.EU_GEO_IDS:
        df = read(geo_id)
        X, y = create_features(df)
        models(X, y)


if __name__ == '__main__':
    main()

