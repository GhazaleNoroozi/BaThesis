import pandas as pd
import config
import matplotlib
from models.arima import Arima
from matplotlib import pyplot as plt
from models.poly_regression import PolyRegression
from models.rf_regression import RandomForest
from models.svr import Svr
from models.sarima import Sarima
from models.ensemble import Ensemble
import csv
from models import model


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
    geo_df.sort_index(inplace=True)
    X: pd.DataFrame = pd.DataFrame(geo_df.index)
    y: pd.Series = geo_df.iloc[:, 3 if config.CASE_OR_DEATH == 'c' else 4]
    y: pd.DataFrame = pd.DataFrame(y)
    return X, y


def visualize_data(X, y, geo_id):
    plt.plot(X, y, color='blue', label='New Cases')
    plt.xticks(rotation=config.ROTATION)
    plt.xlabel('Date')
    plt.ylabel('New Cases' if config.CASE_OR_DEATH == 'c' else 'New Deaths')
    plt.legend(loc='best')
    plt.title(geo_id)   # + '\n RMSE: ' + str(rmse) + ' MAPE: ' + str(mape))
    # plt.savefig(geo_id + '.png')
    plt.clf()
    return


def models(X, y, df, geo_id):
    rmse_list = []
    mape_list = []
    if config.MODEL['arima'] == 1:
        rmse = 0
        rmse, mape, print_me = \
        Arima(geo_id).arima(X, y)
        rmse_list.append(rmse)
        mape_list.append(mape)

    if config.MODEL['arima'] == 1:
        rmse = 0
        rmse, mape, print_me = \
        Sarima(geo_id).sarima(X, y)
        rmse_list.append(rmse)
        mape_list.append(mape)

    if config.MODEL['poly reg'] == 1:
        rmse = 0
        # PolyRegression().pr_diffed(df)
        rmse, mape = \
        PolyRegression(geo_id).polynomial_regression(X, y)
        rmse_list.append(rmse)
        mape_list.append(mape)

    if config.MODEL['rf reg'] == 1:
        rmse, mape =\
            RandomForest(geo_id).rf_reg_diffed(df)
        # rmse, mape = \
        # RandomForest().rf_regression(X, y, df, geo_id)
        # RandomForest().cross_valid(X, y)
        rmse_list.append(rmse)
        mape_list.append(mape)

    if config.MODEL['svr'] == 1:
        rmse = 0
        # Svr().svr_diffed(df, geo_id)
        rmse, mape, pred = \
        Svr(geo_id).svr(X, y)
        # Svr().cross_valid(X, y)
        rmse_list.append(rmse)
        mape_list.append(mape)

    return rmse_list, mape_list


def main():

    matplotlib.rc('xtick', labelsize=8)
    ml = []
    rml = []
    for geo_id in config.EU_GEO_IDS:
        df = read(geo_id)
        X, y = create_features(df)
        #
        df.sort_index(inplace=True)
        geo_df = df.iloc[:, :6]
        XX = geo_df.index
        yy = geo_df.iloc[:, 3 if config.CASE_OR_DEATH == 'c' else 4]
        e = Ensemble(geo_id)
        # e.makeDataset(pd.DataFrame.from_dict({'Date': XX.values, 'Cases': yy.values}), X, y)
        rm, m = e.Ensemble()
        ml.append(m)
        rml.append(rm)
    print(f'{sum(ml)/len(ml)},{sum(rml)/len(rml)}')
    return

    f = open('rmse and mape tables/Prediction_rmse.csv', 'a')
    ff = open('rmse and mape tables/Prediction_mape.csv', 'a')

    # for p in range(0, 7):
    #     for q in range(0, 7):
    #         for d in range(1, 2):
    #             listt = []
    #             for P in range(0, 1):
    #                 for Q in range(0, 7):
    #                     for D in range(1, 2):
    #                         config.SARIMA_TREND = (p, d, q)
    #                         config.SARIMA_SEASONAL = (P, D, Q, 7)
    #                         print(p, d, q, P, D, Q)
    #                         if p == 3 and q == 4:   # and D == 1 and p ==1 and q==3 and d==1:
    #                             continue
    # listt = []
    # for i in range(2, 21):
    for i in range(0, 1):
        # config.PR_DIMENSION = i
        list = []
        for geo_id in config.EU_GEO_IDS:
            df = read(geo_id)
            X, y = create_features(df)

            df.sort_index(inplace=True)
            geo_df = df.iloc[:, :6]
            XX = geo_df.index
            yy = geo_df.iloc[:, 3 if config.CASE_OR_DEATH == 'c' else 4]
            # print(XX.values)
            # print(yy.values)
            # print("______________________________")

            # visualize_data(X, y, geo_id)
            # print(geo_id)
            # print(X.values)
            # print(X.values.tolist())
            rmse_list, mape_list = models(X, y, pd.DataFrame.from_dict({'Date': XX.values, 'Cases': yy.values}), geo_id)
            f.write(f'{geo_id},{rmse_list[0]},{rmse_list[1]},{rmse_list[2]},{rmse_list[3]},{rmse_list[4]}\n')
            ff.write(f'{geo_id},{mape_list[0]},{mape_list[1]},{mape_list[2]},{mape_list[3]},{mape_list[4]}\n')
            # list.append(rmse_list[0])

            # f = open('/SVR_EBV.csv', 'a')
            # f.write(f'Average,{"{:.2f}".format(model.mse1/model.count1)},{"{:.2f}".format(model.bias1/model.count1)},'
            #         f'{"{:.2f}".format(model.var1/model.count1)}\n')
        # mean = sum(list)/len(list)
    f.close()
    ff.close()
        # listt.append(mean)

                            # f.write(f'SARIMA({p}{d}{q})({P}{D}{Q})7,' + '%.2f\n' % mean)
                            # print(f'SARIMA({p}{d}{q})({P}{D}{Q})7,' + '%.2f' % mean)
                            # listt.append(mean)
                            # print(mean)
    # f.close()


# print(f'({p},{d},{q})')
#     print(listt)
#     plt.plot(range(2, 21), listt)
#     plt.xlabel('Order')
#     plt.ylabel('RMSE')
#     plt.show()


if __name__ == '__main__':
    main()
