import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA


data = pd.read_csv('./monthly-car-sales.csv', index_col=['Month'], parse_dates=['Month'])
print(data.head())
print(len(data['Sales']))

decomp = seasonal_decompose(x=data, model='additive')
est_trend = decomp.trend
est_seasonal = decomp.seasonal
est_residual = decomp.resid

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

est_residual.plot(kind='kde')
plt.show()

data_array = np.array(data['Sales'])
partitions = np.split(data_array, indices_or_sections=4)
print("Mean of Partitions")
print(np.mean(partitions, axis=1))
print("Variance of Partitions")
print(np.var(partitions, axis=1))

plt.plot(np.mean(partitions, axis=1), linestyle = ' ', marker = 'o')
plt.show()
plt.plot(np.var(partitions, axis=1), linestyle = ' ', marker = 'o')
plt.show()

# P-Value > 0.05 – Null hypothesis accepted and Time Series is Non- Stationary.
# ADF value is greater than all threshold values of 0.10, 0.05, 0.01
# Therefore our Time-Series data is Non-Stationary
# ADF shoes heteroscedasticity (should be below about 3.4, there is a chart)
# p-value shows trend (should be below 0.05)
adf, pvalue, usedlag, nobs, critical_values, icbest = adfuller(data_array)
print('adf - ', adf)
print('pvalue - ', pvalue)

# Use the period argument when data is not a pandas df
# Period is taken as 12 as its monthly dat and there are 12 months
ss_decomposition = seasonal_decompose(x=data_array, model='additive',period=12)
est_trend2 = ss_decomposition.trend
est_seasonal2 = ss_decomposition.seasonal
est_residual2 = ss_decomposition.resid

adf2, pvalue2, usedlag2, nobs2, critical_values2, icbest2 = adfuller(est_residual2[6:-6])
print('adf - ', adf2)
print('pvalue - ', pvalue2)

log_data = np.log(data_array)
adf3, pvalue3, usedlag3, nobs3, critical_values3, icbest3 = adfuller(log_data)
print('adf - ', adf3)
print('pvalue - ', pvalue3)

model = ARIMA(data_array[6:-6], order=(10,1,0))
model_fit = model.fit(disp=0)
print(model_fit.summary())

