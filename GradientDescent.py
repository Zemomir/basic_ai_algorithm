def f(x):
    return x**4 - 5*x**3+6*x**2-x
def df(x):
    return 4*x**3 -15*x**2+12*x-1
lr = 0.001

x=3.7
epoch = 1000
for i in range(epoch):
    x= x-df(x)*lr
print(x)