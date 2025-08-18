import pandas as pd
df = pd.read_csv(r'C:\Users\hero\Downloads\titanic.csv')
print(df.columns)
df = df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis ='columns')
df['Age']= df['Age'].fillna(df['Age'].mean())
df['Age'] =df['Age'].astype(int)
from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()
df['Nsex'] = le_sex.fit_transform(df['Sex'])
input = df.drop('Sex',axis='columns')
target =df['Survived']
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(input,target)
print("THE SCORE IS", model.score(input, target))
print("\n", model.predict([[3, 22, 1, 0, 0]]))  # Example input for prediction

