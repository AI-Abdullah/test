import pandas as pd
import numpy as np
df= pd.read_excel(r'C:\Users\hero\Desktop\online.xlsx')
print(df)
d=df.describe()
print("AFTER DESCRIBE\n\n", d)
q1 = df['height'].quantile(0.25)
q3 = df['height'].quantile(0.75)
print(q1,"THIS IS SECOND 75 QUANTILE",q3)
IQR = q3 - q1
print("IQR is", IQR)
lower = q1 - 1.5*IQR
upper = q3 + 1.5*IQR
print("lower  bound is =\n", lower, "\nupper bound is =\n", upper)
out = df[(df['height'] < lower) | (df['height'] > upper)]
print("OUTLIER IS\n", out)
nooutlier =df[(df['height']>lower) & (df['height']<upper)]
print("NO OUTLIER IS\n", nooutlier)
