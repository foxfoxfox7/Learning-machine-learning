import pandas as pd
import numpy as np

dff = pd.DataFrame(np.random.randn(10, 3), columns=list('ABC'))
dff.iloc[3:5, 0] = np.nan
dff.iloc[4:6, 1] = np.nan
dff.iloc[5:8, 2] = np.nan

print(dff)

dff = dff.fillna(0)

print(dff)
