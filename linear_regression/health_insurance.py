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

fig, ax = plt.subplots(nrows=2)
sns.distplot(df['charges'], ax=ax[0], hist=False)#.set_title('charges')
sns.distplot(df['log_charges'], ax=ax[1], hist=False)#.set_title('log charges')
plt.show()

# The distibution is skewed so a new column of log charges has been
# created to give a non-skewed data set

print('bmi correlation - ', df['charges'].corr(df['bmi']))
print('age correlation - ', df['charges'].corr(df['age']))

# There is a small but noticable correlation between both age and bmi
# with charges. It is larger with age

sns.boxplot(data=df, x='smoker', y='charges')
plt.show()

# There is a very large difference in the charges for smokers and
# non smokers

sns.lmplot(data=df, x='bmi', y='charges', hue='smoker')
plt.show()

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

fig, ax = plt.subplots(nrows=2)
sns.distplot(df[(df['smoker'] == 'yes')]["charges"], ax=ax[0], hist=False)
sns.distplot(df[(df['smoker'] == 'no')]["charges"], ax=ax[1], hist=False)
plt.show()

# We can see that amount smokers, there are two groups of people. From
# the lmplot of bmi and age there are two groups. Maybe smokers are
# grouped into two differnt groups according to bmi

df['bmi_cat'] = np.nan
df.loc[df['bmi'] < 30, 'bmi_cat'] = 0
df.loc[df['bmi'] > 30, 'bmi_cat'] = 1

sns.lmplot(data=df[df['smoker'] == 'yes'], x='bmi', y='charges', hue='bmi_cat')
plt.show()

# There is a clear differnce in the charges for under and over bmi=30
# Though the gradient remains the same, there is an additional +charge
# once over the bmi=30 threshold

print(df['region'].unique())
sns.boxplot(data=df, x='region', y='charges')
plt.show()
sns.boxplot(data=df, x='sex', y='charges')
plt.show()
sns.boxplot(data=df, x='children', y='charges')
plt.show()

# I can see very little difference in the charges for any of these
# caragorical categories. We will check with Pearson R coef.

# Discrete but ordinal (children) and continuous (bmi, age) data are
# treated the same
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

df_corr = df.drop(['log_charges', 'bmi_cat'], axis = 1)
corr = df_corr.corr()
sns.heatmap(corr, cmap = 'Wistia', annot= True)
plt.show()

# checking for correlations between any of the features
# there is a slight correlation between age and dmi but it can be
# ignored. there is a significant corelation between southeast and bmi

df = pd.read_csv('insurance.csv')
print(df.head())

#df['bmi_cat'] = np.nan
#df.loc[df['bmi'] <= 30, 'bmi_cat'] = 0
#df.loc[df['bmi'] > 30, 'bmi_cat'] = 1

instances_tot = df.shape[0]
print(instances_tot)

df_edit = pd.DataFrame()
bias = [1] * instances_tot
df_edit['bias'] = bias
df_edit['smoker'] = df['smoker']
df_edit['bmi'] = df['bmi']
df_edit['age'] = df['age']
df_edit['age^2'] = df['age'] ** 2
df_edit['log_charges'] = np.log(df['charges'])

df_edit['bmi_cat'] = np.nan
df_edit.loc[df_edit['bmi'] <= 30, 'bmi_cat'] = 0
df_edit.loc[df_edit['bmi'] > 30, 'bmi_cat'] = 1

df_edit['smoker'].replace(('yes', 'no'), (1, 0), inplace=True)
df_edit['age'] = df_edit['age'] / (df_edit['age'].max() - df_edit['age'].min())
df_edit['age^2'] = df_edit['age^2'] / (df_edit['age^2'].max() - df_edit['age^2'].min())
df_edit['bmi'] = df_edit['bmi'] / (df_edit['bmi'].max() - df_edit['bmi'].min())

df_inputs = df_edit.drop('log_charges', axis=1)
df_targets = pd.DataFrame()
df_targets['targets'] = df_edit['log_charges']


def linear_func_mv(x, weights):
    return  x @ weights.t()
    #return np.multiply(weights.t(), x)

def cost_func_mv(X, y, weights):
    mm = len(X)
    return (1/2*mm) * torch.sum((linear_func_mv(X, weights) - y)**2)

def mv_linear_regression_alg(inputs, targets, loops, alpha):

    # require_grad = True in order to backwards compute derivatives of
    # the weights and biases
    # number of weight coefficients equal to the number of features
    w = torch.randn(1, inputs.shape[1], requires_grad=True)

    print('initial cost - ', cost_func_mv(inputs, targets, w))
    print('Improving parameters...')

    cost_tracker = []

    t_mv1 = time.time()
    for i in range(loops):
        cost = cost_func_mv(inputs, targets, w)
        cost_tracker.append(cost_func_mv(inputs, targets, w).detach().numpy())
        cost.backward()
        with torch.no_grad():
            w -= w.grad * alpha
            w.grad.zero_()
    t_mv2 = time.time()

    print(f'...time taken is {t_mv2 - t_mv1} s to complete {loops} loops')
    print('final cost - ', cost_func_mv(inputs, targets, w))

    return w, cost_tracker

########################################################################
# ALL
########################################################################

inputs = torch.tensor(df_inputs.values).float()
targets = torch.tensor(df_targets.values).float()
loops = 10000
alpha = 3e-7
w, costs = mv_linear_regression_alg(inputs, targets, loops, alpha)

print('\nRESULTS\n')
#print(costs)
plt.plot(costs)
plt.show()
print(df_inputs.columns.tolist())
print(w)
df_targets['results'] = linear_func_mv(inputs, w).detach().numpy()
print(df_targets.head())
print(df_targets['targets'].corr(df_targets['results']))

plt.scatter(df_targets['targets'].values, df_targets['results'].values)
plt.title('Correlation between calculated and target values')
plt.xlabel('Target values')
plt.ylabel('Calculated values')
plt.show()

########################################################################
# SMOKERS
########################################################################
df_smokers_i = df_edit[df_edit['smoker'] == 1]
df_smokers_t = pd.DataFrame()
df_smokers_t['targets'] = df_smokers_i['log_charges']
df_smokers_i = df_smokers_i.drop(['log_charges', 'smoker'], axis=1)

inputs_s = torch.tensor(df_smokers_i.values).float()
targets_s = torch.tensor(df_smokers_t.values).float()
loops = 10000
alpha = 3e-7
w_s, costs_s = mv_linear_regression_alg(inputs_s, targets_s, loops, alpha)

print('\nSMOKER RESULTS\n')
#print(costs)
plt.plot(costs_s)
plt.show()
print(df_smokers_i.columns.tolist())
print(w_s)
df_smokers_t['results'] = linear_func_mv(inputs_s, w_s).detach().numpy()

########################################################################
# NON SMOKERS
########################################################################

df_nonsmokers_i = df_edit[df_edit['smoker'] == 0]
df_nonsmokers_t = pd.DataFrame()
df_nonsmokers_t['targets'] = df_nonsmokers_i['log_charges']
df_nonsmokers_i = df_nonsmokers_i.drop(['log_charges', 'smoker'], axis=1)

inputs_ns = torch.tensor(df_nonsmokers_i.values).float()
targets_ns = torch.tensor(df_nonsmokers_t.values).float()
loops = 10000
alpha = 3e-7
w_ns, costs_ns = mv_linear_regression_alg(inputs_ns, targets_ns, loops, alpha)

print('\nNON SMOKER RESULTS\n')
#print(costs)
plt.plot(costs_ns)
plt.show()
print(df_nonsmokers_i.columns.tolist())
print(w_ns)
df_nonsmokers_t['results'] = linear_func_mv(inputs_ns, w_ns).detach().numpy()

########################################################################
# COMBINE
########################################################################

df_final_i = pd.concat([df_nonsmokers_i, df_smokers_i])
df_final_i = df_final_i.sort_index()
df_final_t = pd.concat([df_nonsmokers_t, df_smokers_t])
df_final_t = df_final_t.sort_index()
print(df_final_t.head())
print(df_final_t['targets'].corr(df_final_t['results']))

plt.scatter(df_final_t['targets'].values, df_final_t['results'].values)
plt.title('Correlation between calculated and target values')
plt.xlabel('Target values')
plt.ylabel('Calculated values')
plt.show()
