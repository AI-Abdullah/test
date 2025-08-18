import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()
import matplotlib.pyplot as plt
plt.gray()
for i in range(3):
    plt.matshow(digits.images[i])
    #plt.show() 
df =pd.DataFrame(digits.data)
df['target']= digits.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(df.drop(['target'],axis ='columns'),digits.target,test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50)
model.fit(x_train,y_train)
print("this is score of the model",model.score(x_test,y_test))
y_predicted = model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
import seaborn as sns
sns.heatmap(cm,annot=True)
plt.xlabel('predicted')
plt.ylabel('truth')
plt.show()