#Robust Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\emir\Desktop\projeler veri setleri\zart zurt projelerim\Salary Data.csv")

x = data["YearsExperience"]
y = data["Salary"]

x = np.array(x)
y = np.array(y)
epoch = 25000
delta = 105
w = 1
b = np.random.uniform()
lr = 0.001
for i in range(epoch):
    y_hat =w*x + b
    error = np.abs(y - y_hat)

    mse =0.5*(error**2)
    mae = delta*(np.abs(error)-0.5*delta)

    huber_loss = np.where(error <= delta,mse,mae).mean()
    print(f"huber Loss = {huber_loss}")
    dw = np.where(error <= delta ,-error*x,-delta*np.sign(error)*x).mean()
    db = np.where(error <= delta,-error,delta *np.sign(error)).mean()

    w -= dw*lr
    b -= db*lr
plt.scatter(x,y)
plt.plot(x,y_hat)
plt.show()