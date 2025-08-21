import pandas as pd 
import numpy as np 
import seaborn as sn 
import matplotlib.pyplot as plt 
import warnings 
warnings.filterwarnings('ignore')
dataset = pd.read_csv(r'C:\Users\hero\Downloads\Melbourne_housing_FULL.csv (1)\Melbourne_housing_FULL.csv')
#print(dataset.nunique())
col_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 
               'Distance', 'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset= dataset[col_to_use]
#print(dataset)
#print(dataset.isna().sum())
col_to_fill = ['Propertycount','Bedroom2','Car','Bathroom']
dataset[col_to_fill] =dataset[col_to_fill].fillna(0)
#print(dataset.isna().sum())
dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] =dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())
#print(dataset.isna().sum())
dataset.dropna(inplace=True)
#print(dataset.isna().sum()) to remove all null from all columns
#print(dataset)
dataset= pd.get_dummies(dataset,drop_first=True) # to make all text columns into dummy
#print(dataset)
x= dataset.drop('Price',axis=1)
y=dataset['Price']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=2)
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(x_train,y_train) 
print("Trained data score Before",reg.score(x_train,y_train))
print("Test data score Before",reg.score(x_test,y_test))
from sklearn import linear_model
model = linear_model.Lasso(alpha =50,max_iter=100,tol =0.1)
model.fit(x_train,y_train)
print("Trained data score After",model.score(x_train,y_train))
print("Test data score After",model.score(x_test,y_test))
