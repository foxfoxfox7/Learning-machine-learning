import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import time

import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('medical_no_show.csv', parse_dates=[3,4])
print(df.info())
#print(df.head())

print(df.describe())

# Min age is -1
df = df.loc[df['Age'] >= 0]

print(df['Handcap'].value_counts())

neigh_size = df['Neighbourhood'].unique().size
neighbourhoods = df['Neighbourhood'].unique()
print(df['Neighbourhood'].unique())
print(f'\nNumber of neighbourhoods - {neigh_size}')

print('no show ratio - ', (df['No-show'] == 'Yes').sum() / df.shape[0])
for nn in neighbourhoods:
    num_in_neigh = df[df['Neighbourhood'] == nn].shape[0]
    num_yes_neigh = (df[(df['Neighbourhood'] == nn) & (df['No-show'] == 'Yes')]).shape[0]
    print(nn, ' --- ', num_yes_neigh / num_in_neigh),

# NEED TO CHECK FOR STATISTICAL SIGNIFICANCE
print(df['Neighbourhood'].value_counts())



#df.groupby(df["ScheduledDay"].dt.week)["ScheduledDay"].count().plot(kind="bar", color='b', alpha=0.3)
#plt.show()
#df.groupby(df["AppointmentDay"].dt.week)["AppointmentDay"].count().plot(kind="bar", color='r', alpha=0.3)
#plt.show()


###############################################################################
# FEATURE CREATION
###############################################################################

df['Gender'].replace(('M', 'F'), (1, 0), inplace=True)

dummies = pd.get_dummies(df['Handcap'])
dummies.columns = ['handicap-' + str(col) for col in dummies.columns]
df = pd.concat((df, dummies), axis = 1)
df = df.drop('Handcap', axis = 1)

df['newborn'] = np.NaN
df.loc[df['Age'] > 0, 'newborn'] = 0
df.loc[df['Age'] == 0, 'newborn'] = 1



# I want to find out the number of appointments made by the patient
# BEFORE the current one is made. I also want the number of missed
# appointments BEFORE the current booking
# I will be splitting up the dataframe so I will fix the appointment ID
# as the index so we can recombine

df.set_index('AppointmentID', inplace=True)
df.sort_values(by='ScheduledDay', inplace=True)

df['book_count'] = df.groupby('PatientId').cumcount()


def count_missed_apts_before_now(row, df):
    subdf = df.query("AppointmentDay<@row.ScheduledDay and\
                     `No-show`=='Yes' and PatientId==@row.PatientId")
    return len(subdf)


t3 = time.time()
df['miss_count'] = df.apply(count_missed_apts_before_now, axis=1, args = (df,))
t4 = time.time()
miss_count_t = t4-t3
print(f'miss count column calculated in {miss_count_t}')

print(df.tail(50))
print(df.info())



