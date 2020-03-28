import pandas as pd
from sklearn.linear_model import LinearRegression


df = pd.read_csv("./datasets/houses.csv",sep=";");

x = df.meter.values.reshape(-1,1) #Meter values
y = df.price.values.reshape(-1,1) #Price values

linear_reg = LinearRegression()

linear_reg.fit(x,y)

b0 = linear_reg.intercept_
b1 = linear_reg.coef_
print(b0+b1*300) # 300m = 30k  
