
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    P = np.array([0.2,0.3,0.4,0.9,0.7,1,1,0.36,0.123,0.14])

    return x * P


X = [1,1,1,1,0,1,1,0,1,1]



Y = (f(X))
print(Y)


plt.plot(X, Y, 'g^')
plt.plot(X, Y)
plt.show()

def fobj():
  value = 0
  P = np.array([0.2, 0.3, 0.4, 0.9, 0.7, 1, 1, 0.36, 0.123, 0.14])
  X = np.array([1, 1, 1, 1, 0, 1, 1, 0, 1, 1])
  for i in range(0,len(X)):
      value +=  P[i]*X[i]
  return value

v = fobj()
print(v)

