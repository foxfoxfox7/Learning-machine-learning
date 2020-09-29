import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('insurance.csv')
print(df.head())
print(df.info())
print(df.isnull().sum())

df['log_charges'] = np.log(df['charges'])

#fig, ax = plt.subplots(nrows=2)
#sns.distplot(df['charges'], ax=ax[0], hist=False)#.set_title('charges')
#sns.distplot(df['log_charges'], ax=ax[1], hist=False)#.set_title('log charges')
#plt.show()

# The distibution is skewed so a new column of log charges has been
# created to give a non-skewed data set

print('bmi correlation - ', df['charges'].corr(df['bmi']))
print('age correlation - ', df['charges'].corr(df['age']))

# There is a small but noticable correlation between both age and bmi
# with charges. It is larger with age

#sns.boxplot(data=df, x='smoker', y='charges')
#plt.show()

# There is a very large difference in the charges for smokers and
# non smokers

#sns.lmplot(data=df, x='bmi', y='charges', hue='smoker')
#plt.show()

# when separated into smokers and non-smokers, we see there is a
# much stronger correlation between bmi and charges for smokers
# and an almost non existent one for non-smokers
# These groups are evident in the spread of the data

print('bmi correlation (smokers) - ', df[df['smoker'] == 'yes']['charges'].corr(df['bmi']))
print('bmi correlation (non-smokers) - ', df[(df['smoker'] == 'no')]['charges'].corr(df['bmi']))
print('age correlation (smokers) - ', df[df['smoker'] == 'yes']['charges'].corr(df['age']))
print('age correlation (non-smokers) - ', df[(df['smoker'] == 'no')]['charges'].corr(df['age']))

# For age, there is a stronger correlation between age and charges for
# non smokers (though both are significant)

#fig, ax = plt.subplots(nrows=2)
#sns.distplot(df[(df['smoker'] == 'yes')]["charges"], ax=ax[0], hist=False)
#sns.distplot(df[(df['smoker'] == 'no')]["charges"], ax=ax[1], hist=False)
#plt.show()

# We can see that amount smokers, there are two groups of people. From
# the lmplot of bmi and age there are two groups. Maybe smokers are
# grouped into two differnt groups according to bmi

df['bmi_cat'] = np.nan
df.loc[df['bmi'] < 30, 'bmi_cat'] = 0
df.loc[df['bmi'] > 30, 'bmi_cat'] = 1

#sns.lmplot(data=df[df['smoker'] == 'yes'], x='bmi', y='charges', hue='bmi_cat')
#plt.show()

# There is a clear differnce in the charges for under and over bmi=30
# Though the gradient remains the same, there is an additional +charge
# once over the bmi=30 threshold

#print(df['region'].unique())
#sns.boxplot(data=df, x='region', y='charges')
#plt.show()
#sns.boxplot(data=df, x='sex', y='charges')
#plt.show()
#sns.boxplot(data=df, x='children', y='charges')
#plt.show()

# I can see very little difference in the charges for any of these
# caragorical categories. We will check with Pearson R coef.

# Discrete (children) and continuous (bmi, age) data are treated the same
# Catagorical data (smoker, sex, region) need to be converted to ints
# smoker and sex have only two catagories and so can be converted
# to 0s and 1s
# region has 4 catagories and so can be converted to 0s and 1s with
# dummies

df['smoker'].replace(('yes', 'no'), (1, 0), inplace=True)
df['sex'].replace(('male', 'female'), (1, 0), inplace=True)

dummies = pd.get_dummies(df['region'])
df = pd.concat((df, dummies), axis = 1)
df = df.drop('region', axis = 1)

print('sex correlation - ', df['charges'].corr(df['sex']))
print('children correlation - ', df['charges'].corr(df['children']))
print('smoker correlation - ', df['charges'].corr(df['smoker']))
print('northeast correlation - ', df['charges'].corr(df['northeast']))
print('northwest correlation - ', df['charges'].corr(df['northwest']))
print('southeast correlation - ', df['charges'].corr(df['southeast']))
print('southwest correlation - ', df['charges'].corr(df['southwest']))

#df_corr = df.drop(['log_charges', 'bmi_cat'], axis = 1)
#corr = df_corr.corr()
#sns.heatmap(corr, cmap = 'Wistia', annot= True)
#plt.show()

# checking for correlations between any of the features
# there is a slight correlation between age and dmi but it can be
# ignored. there is a significant corelation between southeast and bmi

#Now I will demonstrate linear regression with just one variable from
# scratch. In this case we will choose a feature that we know has good
# correlation with charges. We will therefore choose age for non-smokers

df_ulr = df[df['smoker'] == 0]
df_ulr = df_ulr.drop(['sex', 'bmi', 'children', 'smoker', 'northeast',
    'northwest', 'southeast', 'southwest', 'bmi_cat'], axis = 1)
print(df_ulr.head())
print(df_ulr.info())

def linear_func(X, th0, th1):
    return (X * th1) + th0

def cost_func(X, y, th0, th1):
    mm = len(X)
    return (1/2*mm) * np.sum(linear_func(X, th0, th1) - y)**2

def grad_descent_th0(X, y, alpha):
    mm = len(X)
    return - (alpha / mm) * np.sum(linear_func(X, th0, th1) - y)

def grad_descent_th1(X, y, alpha):
    mm = len(X)
    return - (alpha / mm) * np.sum((linear_func(X, th0, th1) - y)*X)
