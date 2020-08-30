import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error

data = pd.read_csv('./Datasets/airline-passengers.csv',
    index_col=['Month'], parse_dates=['Month'])
print(data.head())
print(len(data['Passengers']))
data['Passengers'].plot()
plt.show()


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

def test_ARIMA_model(data, order_v, s_order_v = False, split_n = 0.7, show = False):

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
        if s_order_v:
            model4 = ARIMA(expected, order=order_v, seasonal_order=s_order_v)
        else:
            model4 = ARIMA(expected, order=order_v)
        model_fit4 = model4.fit()
        if (t+1) == n_loops:
            print(model_fit4.summary())
        forecast = model_fit4.forecast()
        if show:
            print('forecast - ', forecast[0])
            print('expected - ', test[t])
        predictions = np.append(predictions, forecast[0])
        expected = np.append(expected, test[t])

    return expected, predictions, n_loops

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

def normalize_series(series, norm = 1):

    s_min = np.min(series)
    s_max = np.max(series)

    n_series = [(((x-s_min) / (s_max - s_min)) * norm) for x in series]

    norm_series = np.array(n_series)

    return norm_series

correlation_plots(data['Passengers'], lag = 60)

values = data['Passengers'].values

ss_decomposition = seasonal_decompose(x=values, model='additive',period=12)
est_trend = ss_decomposition.trend
est_seasonal = ss_decomposition.seasonal
est_residual = ss_decomposition.resid

fig, axes = plt.subplots(4, 1)#, figsize=(15, 7))
axes[0].plot(values, label='Original')
axes[0].legend()
axes[1].plot(est_trend, label='Trend',color="b")
axes[1].legend()
axes[2].plot(est_seasonal, label='Seasonality',color='r')
axes[2].legend()
axes[3].plot(est_residual, label='Residuals',color='g')
axes[3].legend()
plt.show()

sns.distplot(est_residual)
plt.show()

#model3 = ARIMA(data['Passengers'], order=(40,1,0))#, seasonal_order=s_order_v
#model_fit3 = model3.fit()#disp = 0
#print(model_fit3.summary())

#model4 = ARIMA(data['Passengers'], order=(10,1,0), seasonal_order=(1,0,0,12))#, seasonal_order=s_order_v
#model_fit4 = model4.fit()#disp = 0
#print(model_fit4.summary())

expected, predicted, n_loops = test_ARIMA_model(data['Passengers'],
 order_v = (4,0,0), s_order_v = (1,1,0,12))#, s_order_v = (1,1,0,12)
dates = data.index.tolist()

expected_n = normalize_series(expected)
predicted_n = normalize_series(predicted)

plot_prediction(expected_n, predicted_n, dates, (2*n_loops))
test_model_fit(expected_n, predicted_n, n_loops)


data['trend_adj'] = data['Passengers'] - est_trend
data = data.dropna()
dates_drop = data.index.tolist()

correlation_plots(data['trend_adj'], lag = 120)

#model5 = ARIMA(data['trend_adj'], order=(2,0,0), seasonal_order=(1,0,0,12))#, seasonal_order=s_order_v
#model_fit5 = model5.fit()#disp = 0
#print(model_fit5.summary())

expected_a, predicted_a, n_loops_a = test_ARIMA_model(data['trend_adj'],
 order_v = (2,0,0), s_order_v = (1,1,0,12))

expected_a_n = normalize_series(expected_a)
predicted_a_n = normalize_series(predicted_a)

plot_prediction(expected_a_n, predicted_a_n, dates_drop, (2*n_loops))
test_model_fit(expected_a_n, predicted_a_n, n_loops)




#values_adj = data['trend_adj']



#model4 = ARIMA(data['Passengers'], order=(40,0,0))#, seasonal_order=s_order_v
#model_fit4 = model4.fit()#disp = 0
#print(model_fit4.summary())
