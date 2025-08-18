import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
df=pd.read_excel(r'C:\Users\hero\Desktop\Book1.xlsx')
print(df)
df.plot(x='Company', y='Revanue',kind='bar')
plt.show()
df.plot(x='Company',y='Revanue',kind='bar',logy=True)
plt.show()