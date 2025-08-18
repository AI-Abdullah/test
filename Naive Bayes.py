import pandas as pd
df = pd.read_csv(r'C:\Users\hero\Downloads\titanic (1).csv')

df.drop(['PassengerId','Name','SibSp','Ticket','Parch','Cabin','Embarked'],axis = 'columns',inplace= True )
target = df.Survived
input =df.drop('Survived',axis ='columns')
dummies = pd.get_dummies(input.Sex ,dtype=int)
input = pd.concat([input,dummies],axis='columns')
#print(input.head(5))
input.drop('Sex', axis='columns',inplace=True)
print(input.columns[input.isna().any()])
input.Age =input.Age.fillna(input.Age.mean())
# print(input.head(10))
#before train our model we split the data into traning and testing 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(input,target,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)
# print(model.predict(x_test[:10]))
print(model.predict_proba(x_test[:10]))