import pandas as pd


df = pd.read_csv('medical_no_show.csv')

print(df.head())
print(df.info())

print(df['Neighbourhood'].value_counts())
