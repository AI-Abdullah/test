import pandas as pd

# Use the correct path and file extension
df = pd.read_csv(r'C:\Users\hero\Downloads\archive (1)\AB_NYC_2019.csv')

# Display the first few rows to confirm it worked
print(df.head())
print(df.price.describe())
min_threshold,max_threshold=df.price.quantile([0.01,0.999])
print(min_threshold ,max_threshold) 
print(df[df.price<min_threshold])
df2= df[(df.price>min_threshold)&(df.price<max_threshold)]
print(df2.shape) 
print(df2.price.describe())