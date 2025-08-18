import pandas as pd 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
isro = load_iris()
x= isro.data
y= isro.target
df =pd.DataFrame(x,columns=isro.feature_names)
df['target']= y
x_train,x_test,y_train,y_test = train_test_split(df.drop(['target'],axis='columns'),isro.target,test_size=0.2)
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

