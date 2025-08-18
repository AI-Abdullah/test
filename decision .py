import pandas as pd 
df = pd.read_csv(r'C:\Users\hero\Downloads\salaries.csv')
print(df.head(3))
input = df.drop('salary_more_then_100k', axis=1)
target =df['salary_more_then_100k']
# here lale encoding is used to convert categorical data into numerical data
from sklearn.preprocessing import LabelEncoder
le_c = LabelEncoder()
le_j = LabelEncoder()
le_d =LabelEncoder()
input['company_n'] = le_c.fit_transform(input['company'])
input['jon_n'] = le_j.fit_transform(input['job'])
input['degree_n'] = le_d.fit_transform(input['degree'])
inputs = input.drop(['company','job','degree'], axis='columns')
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(inputs, target)
print("THE SCORE IS",model.score(inputs,target))
print("\n",model.predict([[2,2,1]]))


