import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
df= pd.read_csv(r'C:\Users\hero\Downloads\heights.csv')
print(df.head())
# outlier detection and removal  using Standard Deviation,kde= true
print(df.height.describe())
sn.histplot(df.height, kde=True)
plt.show()
mean = df.height.mean()
print("the meani  is =",mean)
std=df.height.std()
print("the standard deviation is =",std)
print(mean-3*std)
print(mean+3*std)
print(df[(df.height <54.82) | (df.height > 78.91)])
print("no outlier is")
df_no_outlier = df[(df.height<77.91) & (df.height>54.82)]
print(df_no_outlier.shape)
#Let's add a new column in our dataframe for this Z score
df['zscore'] = ( df.height - df.height.mean() ) / df.height.std()
print (df.head(5))
#Above for first record with height 73.84, z score is 1.94. This means 73.84 is 1.94 standard deviation away from mean
df.height.mean()
df.height.std()
(73.84-66.37)/3.84
df[df['zscore']>3]