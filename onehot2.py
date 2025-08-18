import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\hero\Downloads\carprices (1).csv')
plt.scatter(df['Mileage'],df['Sell Price($)'])
plt.xlabel("MIleage")
plt.ylabel("Sell Price($)")
plt.show()
x = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']
from sklearn.model_selection  import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=10)
print("\n",x_train)
from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(x_train,y_train)
print("",clf.predict(x_test))
print("\n",clf.score(x_test,y_test))
