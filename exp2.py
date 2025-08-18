import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from word2number import w2n

df = pd.read_csv(r'C:\Users\hero\Downloads\hiring.csv')
df['experience']= df['experience'].fillna('zero')
df['experience'] = df['experience'].apply(w2n.word_to_num)
M = df['test_score(out of 10)'].median()
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(M)
reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']], df['salary($)'])
value_topredict = pd.DataFrame([[12,10,10]],columns=['experience','test_score(out of 10)','interview_score(out of 10)'])
predicted = reg.predict(value_topredict)
print("predicted salary for experience 2 years, test score 9, and interview score 6 is:\n",predicted[0])







#df['experience'] = df['experience'].apply(w2n.word_to_num)
#print("the experience column after conversion is:\n", df['experience'])