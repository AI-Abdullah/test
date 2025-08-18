import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv(r'C:\Users\hero\Downloads\my2.csv - Sheet1.csv')
df.columns = df.columns.str.strip()  # Remove leading and trailing spaces from column names
model = linear_model.LinearRegression()
model.fit(df[['AREA']], df['PRICE'])
value_to_predict = np.array([[3300]])  # Fixed indentation
predicated_price = model.predict(value_to_predict)
print("predicated price for area 3300 is:", predicated_price[0])
print("coeficient =\n",model.coef_)  # Coefficient of the linear regression
print("INTERSEPT =",model.intercept_)  # Intercept of the linear regression
import pickle
with open('model_pkl', 'wb') as file:  # Save the model to a file
    pickle.dump(model, file)
with open('model_pkl', 'rb') as file: 
    mp = pickle.load(file)  # Load the model from the file
    print("this is prediction",mp.predict([[5000]]))
# now form joblib
import joblib
joblib.dump(model,'Model_joblib') # save the model to a file
mp = joblib.load('Model_joblib')  # Load the model from the file
print("this is prediction 2",mp.predict([[5000]]))
