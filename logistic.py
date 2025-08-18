import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\hero\Downloads\insurance_data.csv')
plt.scatter(df.age,df.bought_insurance, color='red', marker='+')
plt.xlabel("Age")
plt.ylabel("Bought Insurance")
plt.show()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df[['age']],df.bought_insurance,train_size=0.9,random_state=10)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train,y_train)
print(model.predict(x_test)) 
print(model.predict([[50]]))