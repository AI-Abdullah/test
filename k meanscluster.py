import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df =pd.read_csv(r'C:\Users\hero\Downloads\income.csv')
plt.scatter(df['Age'], df['Income($)'])
plt.title('AGE VS INCOME')
plt.xlabel('AGE')
plt.ylabel('INCOME($)')
#plt.show()
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
df['cluster'] = y_predicted
df1= df[df.cluster==0]
df2= df[df.cluster==1]
df3= df[df.cluster==2]

plt.scatter(df1.Age,df1['Income($)'], color ='green')
plt.scatter(df2.Age,df2['Income($)'], color ='red')
plt.scatter(df3.Age,df3['Income($)'], color ='black')
plt.xlabel('AGE')
plt.ylabel('INCOME($)')
plt.legend()
scaler = MinMaxScaler()
df[['Income($)', 'Age']] = scaler.fit_transform(df[['Income($)', 'Age']])
# after this
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age','Income($)']])
print(y_predicted)
df['clusters'] = y_predicted
df.drop('cluster', axis=1, inplace=True)
df1 = df[df.clusters==0]
df2 = df[df.clusters==1]
df3 = df[df.clusters==2]
plt.scatter(df1.Age,df1['Income($)'],color='green')
plt.scatter(df2.Age,df2['Income($)'],color='red')
plt.scatter(df3.Age,df3['Income($)'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.legend()
plt.show()
# Elbow Plot
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['Age','Income($)']])
    sse.append(km.inertia_)
    plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()


