import pandas as pd
import numpy as np 

df = pd.read_csv(r'C:\Users\hero\Downloads\homeprices (1).csv')
dummy = pd.get_dummies(df.town, dtype = int)
print(dummy)
merge = pd.concat([df,dummy], axis = 'columns')
Final = merge.drop(['town','west windsor'], axis ='columns')
print("\n",Final)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
X= Final.drop('price', axis = 'columns')
Y = Final.price
model.fit(X,Y)
A= model.predict([[2880,0,1]])
print("Predicted price for area 2880 in west windsor is:", A)
print(df)