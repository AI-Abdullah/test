import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
x = iris.data
y =iris.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=40)
#
from sklearn.model_selection import cross_val_score
scores_L= cross_val_score(LogisticRegression(),x,y,cv=5)
print("logistic cross\n",scores_L)
score_s = cross_val_score(SVC(),x,y,cv=3)
print("SVM cross\n",score_s)
score_D= cross_val_score(RandomForestClassifier(n_estimators=50),x,y,cv=5)
print("Random Forest cross\n",score_D)
scores2 = cross_val_score(RandomForestClassifier(n_estimators=40),x,y, cv=5)
print("AVG OF THIS",np.average(scores2))
