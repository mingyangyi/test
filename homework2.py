import numpy as np
from sympy import *

def BFGS(x_1,x_2,k):
    H=np.eye(2)
    grad=np.asmatrix([[2*x_1],[4*x_2]])
    x=np.asmatrix([[x_1],[x_2]])
    A=np.asmatrix([[2,0],[0,4]])
    for i in range(k):
        if grad.all()!=0:
            p=-H*grad
            a=-((x.T*A*p)/(p.T*A*p))[0,0]
            x_prime=x+a*p
            grad_prime=np.asmatrix([[2*x_prime[0,0]],[4*x_prime[1,0]]])
            s=x_prime-x
            y=grad_prime-grad
            rho=1/((y.T*s)[0,0])
            H=(np.eye(2)-rho*s*y.T)*H*(np.eye(2)-rho*y*s.T)+rho*s*s.T
            x=x_prime
            grad=grad_prime
    return x

#print(BFGS(1.0,1.0,2))

def DFP(x_1,x_2,k):
    H=np.eye(2)
    grad=np.asmatrix([[2*x_1],[4*x_2]])
    x=np.asmatrix([[x_1],[x_2]])
    A=np.asmatrix([[2,0],[0,4]])
    for i in range(k):
        if grad.all()!=0:
            p=-H*grad
            a=-((x.T*A*p)/(p.T*A*p))[0,0]
            x_prime=x+a*p
            grad_prime=np.asmatrix([[2*x_prime[0,0]],[4*x_prime[1,0]]])
            s=x_prime-x
            y=grad_prime-grad
            rho=1/((y.T*s)[0,0])
            H=H-1/((y.T*H*y)[0,0])*H*y*y.T*H+rho*s*s.T
            x=x_prime
            grad=grad_prime
    return x

#print(DFP(1,1,2))

def penality(x_1,x_2,k):
    mu,tau=1,1
    grad=np.asmatrix([[1+2*mu*x_1*(pow(x_1,2)+pow(x_2,2)-2)],[1+2*mu*x_2*(pow(x_1,2)+pow(x_2,2)-2)]])
    x=[]
    for i in range(k):
        while (grad.T*grad)[0,0]>tau:
            x_1=x_1-0.05*(1+2*mu*x_1*(pow(x_1,2)+pow(x_2,2)-2))
            x_2=x_2-0.05*(1+2*mu*x_2*(pow(x_1,2)+pow(x_2,2)-2))
            grad = np.asmatrix([[1 + 2 * mu * x_1 * (pow(x_1, 2) + pow(x_2, 2) - 2)],
                                [1 + 2 * mu * x_2 * (pow(x_1, 2) + pow(x_2, 2) - 2)]])
        x.append([x_1,x_2])
        mu,tau=i+2,1/(i+2)
    return (x)

print(penality(-1,1,10))

x_1,x_2=1,1
mu=1
tau=1
x=[]
grad=np.asmatrix([[1+2*mu*x_1*(pow(x_1,2)+pow(x_2,2)-2)],[1+2*mu*x_2*(pow(x_1,2)+pow(x_2,2)-2)]])
while (grad.T * grad)[0, 0] > tau:
    x_1 = x_1 - (1 + 2 * mu * x_1 * (pow(x_1, 2) + pow(x_2, 2) - 2))
    x_2 = x_2 - (1 + 2 * mu * x_2 * (pow(x_1, 2) + pow(x_2, 2) - 2))
    grad = np.asmatrix([[1 + 2 * mu * x_1 * (pow(x_1, 2) + pow(x_2, 2) - 2)],
                        [1 + 2 * mu * x_2 * (pow(x_1, 2) + pow(x_2, 2) - 2)]])
    x.append((grad.T * grad)[0, 0])
#print(x)