import akshare as ak
import pandas as pd
import pydotplus
from sklearn import tree, svm, preprocessing, linear_model, neural_network
import matplotlib.pyplot as plt
from collections import Counter

store_path = r'E:\Postgraduate\First Year\Stock index forecasting'
columns_name = ['date', 'open', 'high', 'low', 'close', 'volume']


def get_stock_index():
    # 沪深300
    csi300 = ak.stock_zh_index_daily(symbol="sh000300")
    csi300.columns = columns_name
    csi300.loc[:, 'gain'] = (csi300['close'] - csi300['open'])/csi300['open']*100
    print(csi300.shape)
    csi300.to_csv(store_path+r".\csi300.csv", index=False, sep=',')
    # 美国 us500
    us500 = ak.index_investing_global_from_url(
        url="https://www.investing.com/indices/us-spx-500", period="每日", start_date="20060101",
        end_date="20211224")
    us500.columns = columns_name
    us500.loc[:, 'gain'] = (us500['close'] - us500['open']) / us500['open'] * 100
    print(us500.shape)
    us500.to_csv(store_path+r".\us500.csv", index=False, sep=',')
    # 日本 jp225
    jp225 = ak.index_investing_global_from_url(
        url="https://www.investing.com/indices/japan-ni225", period="每日", start_date="20060101",
        end_date="20211224")
    jp225.columns = columns_name
    jp225.loc[:, 'gain'] = (jp225['close'] - jp225['open']) / jp225['open'] * 100
    print(jp225.shape)
    jp225.to_csv(store_path+r".\jp225.csv", index=False, sep=',')
    # 英国 uk100
    uk100 = ak.index_investing_global_from_url(
        url="https://www.investing.com/indices/uk-100", period="每日", start_date="20060101",
        end_date="20211224")
    uk100.columns = columns_name
    uk100.loc[:, 'gain'] = (uk100['close'] - uk100['open']) / uk100['open'] * 100
    print(uk100.shape)
    uk100.to_csv(store_path+r".\uk100.csv", index=False, sep=',')
    # 德国 DAX30
    dax30 = ak.index_investing_global_from_url(
        url="https://www.investing.com/indices/germany-30", period="每日", start_date="20060101",
        end_date="20211224")
    dax30.columns = columns_name
    dax30.loc[:, 'gain'] = (dax30['close'] - dax30['open']) / dax30['open'] * 100
    print(dax30.shape)
    dax30.to_csv(store_path+r".\dax30.csv", index=False, sep=',')
    # 法国 CAC40
    cac40 = ak.index_investing_global_from_url(
        url="https://www.investing.com/indices/france-40", period="每日", start_date="20060101",
        end_date="20211224")
    cac40.columns = columns_name
    cac40.loc[:, 'gain'] = (cac40['close'] - cac40['open']) / cac40['open'] * 100
    cac40.to_csv(store_path+r".\cac40.csv", index=False, sep=',')


def data_cleaning():
    csi300 = pd.read_csv(store_path+r".\csi300.csv", usecols=["date", "gain"])
    csi300.columns = ['date', 'gain_csi300']
    us500 = pd.read_csv(store_path + r".\us500.csv", usecols=["date", "gain"])
    us500.columns = ['date', 'gain_us500']
    jp225 = pd.read_csv(store_path + r".\jp225.csv", usecols=["date", "gain"])
    jp225.columns = ['date', 'gain_jp225']
    uk100 = pd.read_csv(store_path + r".\uk100.csv", usecols=["date", "gain"])
    uk100.columns = ['date', 'gain_uk100']
    dax30 = pd.read_csv(store_path + r".\dax30.csv", usecols=["date", "gain"])
    dax30.columns = ['date', 'gain_dax30']
    cac40 = pd.read_csv(store_path + r".\cac40.csv", usecols=["date", "gain"])
    cac40.columns = ['date', 'gain_cac40']

    data = csi300.merge(us500, on='date')
    print(data.shape)
    data = data.merge(jp225, on='date')
    print(data.shape)
    data = data.merge(uk100, on='date')
    print(data.shape)
    data = data.merge(dax30, on='date')
    print(data.shape)
    data = data.merge(cac40, on='date')
    print(data.shape)
    data.to_csv(store_path + r"\stock_index_gain.csv", index=False, sep=',')


def svr_sif(x_train_, y_train_, x_test_, y_test_):
    kernels_list = ['linear', 'poly', 'rbf']
    colors_list = ['red', 'green', 'blue']
    marks_list = ['o', '*', '^']
    plt.figure(figsize=(20, 10))
    for kernel, color, mark in zip(kernels_list, colors_list, marks_list):
        print(kernel)
        svr = svm.SVR(kernel=kernel)
        svr.fit(x_train_, y_train_)
        predict = svr.predict(x_test_)
        svm_score = svr.score(x_test_, y_test_)
        count = Counter(predict * y_test_ > 0)
        accuracy = count[True] / (count[True] + count[False])
        print(count)
        print('accuracy:', accuracy)
        print('count>0', Counter(y_test_ > 0))
        print("score: ", svm_score)
        pd.DataFrame(predict).to_csv(store_path + '\\' + kernel + "_result.csv", index=False, sep=',')
        plt.scatter(x_test_.index, predict - y_test_, s=5, c=color, marker=mark,
                    label=kernel+':'+str(round(accuracy, 2)))

    plt.axhline(c='black')
    plt.legend()
    plt.title('Prediction Error of SVR using Different Kernel Functions')
    plt.xlabel("Date")
    plt.ylabel("Prediction Error")
    plt.savefig(store_path + r".\svr.png", dpi=600, bbox_inches='tight')
    plt.show()


def lr_sif(x_train_, y_train_, x_test_, y_test_):
    model_list = [linear_model.LinearRegression(), linear_model.Ridge(),
                  linear_model.SGDRegressor(), linear_model.Lasso()]
    name_list = ['LinearRegression', 'RidgeRegressor', 'SGDRegressor', 'LassoRegressor']
    colors_list = ['red', 'green', 'blue', 'black']
    marks_list = ['o', '*', '^', 's']
    plt.figure(figsize=(20, 10))
    for model, color, mark, reg_name in zip(model_list, colors_list, marks_list, name_list):
        print(reg_name)
        reg = model.fit(x_train_, y_train_)
        print("score: ", reg.score(x_test_, y_test_))
        print(reg.coef_)
        predict = reg.predict(x_test_)
        pd.DataFrame(predict).to_csv(store_path + '\\' + reg_name + "_result.csv", index=False, sep=',')
        count = Counter(predict * y_test_ > 0)
        accuracy = count[True] / (count[True] + count[False])
        print(count)
        print('accuracy:', accuracy)
        plt.scatter(x_test_.index, predict - y_test_, s=5, c=color, marker=mark,
                    label=reg_name + ':' + str(round(accuracy, 2)))

    plt.axhline(c='black')
    plt.legend()
    plt.title('Prediction Error of Linear Regression')
    plt.xlabel("Date")
    plt.ylabel("Prediction Error")
    plt.savefig(store_path + r".\lr.png", dpi=600, bbox_inches='tight')
    plt.show()


def dtr_sip(x_train_, y_train_, x_test_, y_test_):
    dtr = tree.DecisionTreeRegressor(random_state=0, max_depth=5)
    dtr = dtr.fit(x_train_, y_train_)
    predict = dtr.predict(x_test_)
    print([x_train_.columns])
    print(tree.export_text(dtr))
    print("测试精度:%f" % (dtr.score(x_test_, y_test_)))
    pd.DataFrame(predict).to_csv(store_path + '\\dtr_result.csv', index=False, sep=',')
    count = Counter(predict * y_test_ > 0)
    accuracy = count[True] / (count[True] + count[False])
    print(count)
    print('accuracy:', accuracy)
    plt.figure(figsize=(20, 10))
    plt.scatter(x_test_.index, predict - y_test_, s=5, c='red', marker='o',
                label='decision tree:'+str(round(accuracy, 2)))

    plt.axhline(c='black')
    plt.legend()
    plt.title('Prediction Error of Decision Trees Regression')
    plt.xlabel("Date")
    plt.ylabel("Prediction Error")
    plt.savefig(store_path + r".\dt.png", dpi=600, bbox_inches='tight')
    plt.show()

    with open(store_path + '\\dtr.dot', 'w') as f:
        f = tree.export_graphviz(dtr,  out_file=f,
                                 filled=True, class_names=True, proportion=True, rounded=True)


def print_predict(x_test_):
    predict = pd.read_csv(store_path + r".\rbf_result.csv")

    plt.figure(figsize=(20, 10))
    plt.scatter(x_test_.index, predict, s=5, c='black', marker='o')

    plt.axhline(c='black')
    plt.title('Predicting Outcomes')
    plt.xlabel("Date")
    plt.ylabel("Predicted Value")
    plt.savefig(store_path + r".\Predicted.png", dpi=600, bbox_inches='tight')
    plt.show()


def nn_sip(x_train_, y_train_, x_test_, y_test_):
    activation_list = ['identity', 'logistic', 'tanh', 'relu']
    colors_list = ['red', 'green', 'blue', 'black']
    marks_list = ['o', '*', '^', 's']
    plt.figure(figsize=(20, 10))
    for act, color, mark in zip(activation_list, colors_list, marks_list):
        print(act)
        nn_reg = neural_network.MLPRegressor(random_state=1, activation=act)
        nn_reg = nn_reg.fit(x_train_, y_train_)
        predict = nn_reg.predict(x_test_)
        pd.DataFrame(predict).to_csv(store_path + '\\nn_result.csv', index=False, sep=',')
        print("score: ", nn_reg.score(x_test_, y_test_))
        count = Counter(predict * y_test_ > 0)
        accuracy = count[True] / (count[True] + count[False])
        print(count)
        print('accuracy:', accuracy)
        plt.scatter(x_test_.index, predict - y_test_, s=5, c=color, marker=mark,
                    label=act + ':' + str(round(accuracy, 2)))

    plt.axhline(c='black')
    plt.legend()
    plt.title('Prediction Error of Neural Network')
    plt.xlabel("Date")
    plt.ylabel("Prediction Error")
    plt.savefig(store_path + r".\nn.png", dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    print('Hello')
    # get_stock_index()
    # data_cleaning()
    # 读取数据
    X = pd.read_csv(store_path+r".\stock_index_gain.csv", parse_dates=['date'])
    X['date'] = pd.to_datetime(X['date'])
    X.set_index('date', inplace=True)
    # 归一化
    # index_X = X.index
    # columns_X = X.columns
    # X = preprocessing.StandardScaler().fit_transform(X)
    # X = pd.DataFrame(X, index=index_X, columns=columns_X)
    # X.to_csv(store_path + r"\data_scaler.csv", index=False, sep=',')
    # 预测数据后一天的沪深300指数
    Y = X.loc['2006-01-05':, 'gain_csi300']
    # 划分训练/测试集
    x_train = X.loc['2006-01-04':'2015-12-31', :]    # 2006-2016
    x_test = X.loc['2016-01-04':'2021-12-22', :]    # 2016-2021
    y_train = Y.loc[:'2016-01-04']
    y_test = Y.loc['2016-01-05':]
    # nn_sip(x_train, y_train, x_test, y_test)
    # print(x_train, y_train, x_test, y_test)
    # svr_sif(x_train, y_train, x_test, y_test)
    # lr_sif(x_train, y_train, x_test, y_test)
    # dtr_sip(x_train, y_train, x_test, y_test)
    #print_predict(x_test)