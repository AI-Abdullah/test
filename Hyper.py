import pandas as pd 
import numpy as np
from sklearn import datasets,svm
irsi = datasets.load_iris()
df = pd.DataFrame(irsi.data,columns=irsi.feature_names)
df['flower']= irsi.target
df['flower']= df['flower'].apply(lambda x:irsi.target_names[x])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(irsi.data,irsi.target,test_size=0.3)
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['rbf','linear']
        }  
    },
    'random_forest': {
        'model': RandomForestClassifier(),
        'params' : {
            'n_estimators': [1,5,10]
        }
    },
    'logistic_regression' : {
        'model': LogisticRegression(solver='liblinear',multi_class='auto'),
        'params': {
            'C': [1,5,10]
        }
    }
}
from sklearn.model_selection import GridSearchCV
scores = []

for model_name, mp in model_params.items():
    clf =  GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
    clf.fit(irsi.data, irsi.target)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })
    
df = pd.DataFrame(scores,columns=['model','best_score','best_params'])
print(df)