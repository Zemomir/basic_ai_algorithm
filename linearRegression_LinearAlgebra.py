import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

data = pd.read_csv(r"Salary Data.csv")
x = data["YearsExperience"]
y = data["Salary"]
n = len(x)
a = np.array([[n,x.sum()],[x.sum(),(x**2).sum()]])
b = np.array([[y.sum()],[(y*x).sum()]])

inv_A = np.linalg.inv(a)
coeff = np.dot(inv_A,b)

b = coeff[0][0]
w = coeff[1][0]


y_hat = w*x + b
plt.scatter(x,y,c ="#fa0")
plt.plot(x,y_hat)
plt.show()