import copy
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error


data = pd.read_csv('./data/BAJAJFINSV.csv',
    index_col=['Date'], parse_dates=['Date'])

#data.Trades.plot()
#plt.show()

data['Trades'] = data['Trades'].fillna(data['Trades'].median())
data = data.drop(['Symbol', 'Series'], axis = 1)

print(data.head())
print(data.info())
print(len(data.VWAP))

data.VWAP.plot()
plt.show()

lag_features = ["High", "Low", "Volume", "Turnover", "Trades"]
exog_feats = ["High", "Low", "Volume", "Turnover", "Trades"]
lag_times = [3, 7, 30]

for feature in lag_features:
    for time in lag_times:
        data[feature+'_ml_'+str(time)] = data[feature].rolling(window=time, min_periods=1).mean()
        data[feature+'_sl_'+str(time)] = data[feature].rolling(window=time, min_periods=1).std()
        exog_feats.append(feature+'_ml_'+str(time))
        exog_feats.append(feature+'_sl_'+str(time))

data.High.plot()
data.High_ml_3.plot()
plt.show()

data = data.fillna(method = 'bfill')
data["month"] = data.index.month
data["week"] = data.index.week
data["day"] = data.index.day
data["day_of_week"] = data.index.dayofweek
data.head()

add_feats = ['month', 'week', 'day', 'day_of_week']
for feat in add_feats:
    exog_feats.append(feat)

print(data.head())
print(data.info())

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

def test_ARIMA_model(data, order_v, split_n = 0.7, show = False):

    #split = int(len(data) * split_n)
    #train, test = data[0:split], data[split:len(data)]

    train, test = train_test_split(data, test_size=0.3, random_state=0)

    expected = copy.copy(train.VWAP)
    predictions = copy.copy(train.VWAP)

    n_loops = len(test)
    if show:
        print(f'{n_loops} loops to calculate')

    for t in range(n_loops):
        if show:
            print('calculating %f loop...' % t)
        model4 = ARIMA(expected, exog = data[exog_feats], order=order_v)#, seasonal_order=s_order_v
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

#correlation_plots(data.VWAP, lag = 1000)

ss_decomposition = seasonal_decompose(x=data.VWAP, model='additive',period=12)
est_trend = ss_decomposition.trend
est_seasonal = ss_decomposition.seasonal
est_residual = ss_decomposition.resid

fig, axes = plt.subplots(4, 1)#, figsize=(15, 7))
axes[0].plot(data.VWAP, label='Original')
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


#expected, predicted, n_loops = test_ARIMA_model(data, order_v = (0,1,4))
#expected_n = normalize_series(expected)
#predicted_n = normalize_series(predicted)


train, test = train_test_split(data, test_size=0.3, random_state=0)
n_periods = len(test)

'''

model = ARIMA(train.VWAP, exog = train[exog_feats], order = (0,1,4))
model_fit = model.fit()
forecast = model_fit.predict(n_periods=n_periods, exog=test[exog_feats])

test["Forecast_ARIMAX"] = forecast.values[-len(test):]
test[["VWAP", "Forecast_ARIMAX"]].plot()
print(test.head())
plt.show()

dates = test.index.tolist()
plot_prediction(test["VWAP"], test["Forecast_ARIMAX"], dates, (1.5*n_periods))
test_model_fit(test["VWAP"], test["Forecast_ARIMAX"], n_periods)


'''

print(exog_feats)
model = auto_arima(train.VWAP, exogenous=train[exog_feats], trace=True,
    error_action="ignore", suppress_warnings=True, start_p=0, start_q=4)
model.fit(train.VWAP, exogenous=train[exog_feats])
print(model.summary())

forecast = model.predict(n_periods=len(test), exogenous=test[exog_feats])
test["Forecast_ARIMAX"] = forecast
test[["VWAP", "Forecast_ARIMAX"]].plot()
plt.show()

expected = np.array(test["VWAP"].values)
predicted = np.array(test["Forecast_ARIMAX"].values)
print(expected[:20])
print(predicted[:20])

dates = data.index.tolist()
#dates = [x for x in range()]

plot_prediction(expected, predicted, dates, (1.5*n_periods))
test_model_fit(expected, predicted, n_periods)
