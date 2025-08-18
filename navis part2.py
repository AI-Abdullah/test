import pandas as pd
df = pd.read_csv(r'C:\Users\hero\Downloads\spam.csv')
print(df.groupby('Category').describe())
df['spam'] = df['Category'].apply(lambda x:1 if x=='spam'else 0)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(df.Message,df.spam,test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
x_train_count =v.fit_transform(x_train.values)
#print(x_train_count.toarray()[:3])
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train_count,y_train)
emails = [
    'Hey mohan, can we get together to watch footbal game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
print(model.predict(emails_count))
# first concert into 
X_test_count = v.transform(x_test)
print(model.score(X_test_count, y_test))
# above methord we are going to use sklear api
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(x_train, y_train)
print(clf.score(x_test,y_test))
clf.predict(emails)