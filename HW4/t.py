import numpy as np

def k(x1, x2):
    return x1*x2
# def k(x1, x2):
    # return (x1*x2)**2+np.e**(x1**2)*np.e**(x2**2)
def kk(x):
    return x**2+np.e**x-1

x1=-0.5
x2=0

K =np.array([[kk(k(x1, x1)), kk(k(x1, x2))],
            [kk(k(x1, x2)), kk(k(x2, x2))]])

K= np.array( [[0,-2], [-2,0]])
print(K)
print(K.shape)
print(np.linalg.det(K))
print(np.linalg.eigh(K))