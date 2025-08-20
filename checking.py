from sklearn import datasets
import pandas as pd
wine = datasets.load_wine()
df = pd.DataFrame(wine.data,columns=wine.feature_names)
df['target']= wine.target
print(df[50:70])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(wine.data,wine.target,test_size=0.3,random_state=100)
from sklearn.naive_bayes import GaussianNB,MultinomialNB
model2 =MultinomialNB()
model = GaussianNB()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
model2.fit(x_train,y_train)
print("this is my multinomail model",model2.score(x_test,y_test))