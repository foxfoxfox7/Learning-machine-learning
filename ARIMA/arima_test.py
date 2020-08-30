import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import chisquare
from sklearn.metrics import mean_squared_error


data = pd.read_csv('./Datasets/monthly-car-sales.csv',
    index_col=['Month'], parse_dates=['Month'])
print(data.head())
print(len(data['Sales']))
values = data['Sales'].values
dates = data.index.tolist()

data['Sales'].plot()
plt.show()

# shows the dependdence of data on previous data
# Meets the significant line between 2 and 3 points so p = 2 or 3
pd.plotting.autocorrelation_plot(data)
plt.show()

adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(values)
print('adf - ', adf)
print('pvalue - ', pvalue)

data['log_values'] = np.log(data['Sales'])
data['log_values'].plot()
plt.show()

ss_decomposition = seasonal_decompose(x=values, model='additive',period=12)
est_trend = ss_decomposition.trend
est_seasonal = ss_decomposition.seasonal
est_residual = ss_decomposition.resid

fig, axes = plt.subplots(4, 1)#, figsize=(15, 7))
axes[0].plot(data, label='Original')
axes[0].legend()
axes[1].plot(est_trend, label='Trend',color="b")
axes[1].legend()
axes[2].plot(est_seasonal, label='Seasonality',color='r')
axes[2].legend()
axes[3].plot(est_residual, label='Residuals',color='g')
axes[3].legend()
plt.show()

data['trend_adj'] = data['Sales'] - est_trend
data['trend_adj'].plot()
plt.show()
data['resids'] = est_residual
data['resids'].plot.hist()
plt.show()

def test_ARIMA_model(data, order_v, s_order_v = (1,1,0,12), split_n = 0.7, show = True):

    split = int(len(data) * split_n)
    train, test = data[0:split], data[split:len(data)]

    expected = copy.copy(train)
    predictions = copy.copy(train)

    n_loops = len(test)
    if show:
        print(f'{n_loops} loops to calculate')

    for t in range(n_loops):
        if show:
            print('calculating %f loop...' % t)
        #model4 = ARIMA(expected, order=order_v)
        model4 = ARIMA(expected, order=order_v, seasonal_order=s_order_v)
        model_fit4 = model4.fit()#disp = 0
        if (t+1) == n_loops:
            print(model_fit4.summary())
        forecast = model_fit4.forecast()
        if show:
            print('forecast - ', forecast[0])
            print('expected - ', test[t])
        predictions = np.append(predictions, forecast[0])
        expected = np.append(expected, test[t])

    return expected, predictions, n_loops#, forecast_error, percent_error


def plot_prediction(expect, predict, time_ax, window):

    win = int(window)
    expected = expect[-win:-1]
    predicted = predict[-win:-1]
    time_axis = time_ax[-win:-1]

    plt.plot(time_axis, expected, label = 'expected')
    plt.plot(time_axis, predicted, label = 'predicted')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def test_model_fit(expect, predict, window):

    win = int(window)
    expected = expect[-win:-1]
    predicted = predict[-win:-1]

    error = expected - predicted
    percentage_error = np.abs(error) / np.abs(expected)

    print('bias - ', np.mean(error))
    print('mean error - ', np.mean(np.abs(error)))
    print('median error - ', np.median(np.abs(error)))
    mse = mean_squared_error(expected, predicted)
    print('rms - ', math.sqrt(mse))
    print('percentage error - ', (np.mean(percentage_error)*100))

def correlation_plots(data, lag = 40):

    lag_acf = acf(data, nlags=lag)
    lag_pacf = pacf(data, nlags=lag)# , method='ols'

    plt.subplot(121)
    plt.plot(lag_acf)
    plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')

    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=-1.96/np.sqrt(len(data)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(data)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.show()


correlation_plots(values)

data['diff'] = data['Sales'].diff()
print(data.head())

correlation_plots(data['diff'][1:])

ss_decomposition_d = seasonal_decompose(x=data['diff'][1:], model='additive',period=12)
est_trend_d = ss_decomposition_d.trend
est_seasonal_d = ss_decomposition_d.seasonal
est_residual_d = ss_decomposition_d.resid

fig, axes = plt.subplots(4, 1)#, figsize=(15, 7))
axes[0].plot(data['diff'][1:], label='Original')
axes[0].legend()
axes[1].plot(est_trend_d, label='Trend',color="b")
axes[1].legend()
axes[2].plot(est_seasonal_d, label='Seasonality',color='r')
axes[2].legend()
axes[3].plot(est_residual_d, label='Residuals',color='g')
axes[3].legend()
plt.show()

expect, predict, predict_n = test_ARIMA_model(
    data = data['diff'][1:], order_v = (2,1,0), show = False)

plot_prediction(expect, predict, dates[1:], (2*predict_n))
test_model_fit(expect, predict, predict_n)





expect, predict, predict_n = test_ARIMA_model(
    data = values, order_v = (3,1,0), show = False)

plot_prediction(expect, predict, dates, (2*predict_n))
test_model_fit(expect, predict, predict_n)



'''
data_drop = data.copy()
data_drop = data_drop.dropna()
adj_values = data_drop['trend_adj'].values
resid_vals = data_drop['resids'].values
dates_drop = data_drop.index.tolist()

correlation_plots(adj_values)

expect_a, predict_a, predict_n_a = test_ARIMA_model(
    data = adj_values, order_v = (2,0,0), show = False)

plot_prediction(expect_a, predict_a, dates_drop, (2*predict_n_a))
test_model_fit(expect_a, predict_a, predict_n_a)

#expect_r, predict_r, predict_n_r = test_ARIMA_model(
#    data = resid_vals, order_v = (4,0,0), show = False)

#plot_prediction(expect_r, predict_r, dates_drop, (2*predict_n_r))
#test_model_fit(expect_r, predict_r, predict_n_r)
'''
