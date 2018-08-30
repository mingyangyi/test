import numpy as np
from sympy import *

#dogleg方法确定迭代方向
def dogleg_direction(x_1,x_2,delta):
    g=np.transpose(np.asmatrix([-400*x_1*(x_2-pow(x_1,2))-2*(1-x_1),200*(x_2-pow(x_1,2))]))
    B=np.asmatrix([[-400*x_2+1200*pow(x_1,2)+2,-400*x_1],[-400*x_1,200]])
    p_u=-((np.transpose(g)*g)/(np.transpose(g)*B*g))[0,0]*g
    p_b=-B.I*g
    if (p_b.T*p_b)[0,0]<delta:
        return p_b
    elif (p_u.T*p_u)[0,0]>delta:
        return delta*p_u/((p_u.T*p_u)[0,0])
    x=symbols('x')
    tau=solve(((p_u+(x-1)*(p_u-p_b)).T*(p_u+(x-1)*(p_u-p_b)))[0,0]-pow(delta,2),x)
    tau=np.asmatrix(tau)
    tau=tau[tau>0]
    return p_u+(tau-1)*(p_u-p_b)

#信赖域方法求迭代
def trust_region(x_1,x_2,delta,eta,k,bound):
    x=[]
    for i in range(k):
        dir = dogleg_direction(x_1,x_2,delta)
        x_1_prime=x_1+dir[0,0]
        x_2_prime=x_2+dir[1,0]
        gra = np.asmatrix([400*x_1*(x_2-pow(x_1,2))+2*(1-x_1),-200*(x_2-pow(x_1,2))]).T
        Hessian=np.asmatrix([[-400*x_2+1200*pow(x_1,2)+2,-400*x_1],[-400*x_1,200]])
        rho_1=100*pow((x_2-pow(x_1,2)),2)+pow((1-x_1),2)-100*pow((x_2_prime-pow(x_1_prime,2)),2)-pow((1-x_1_prime),2)
        rho_2=gra.T*dir-0.5*(dir.T*Hessian*dir)
        rho=rho_1/rho_2
        if rho<1/4:
            delta=1/4*delta
        elif rho>3/4 and (dir.T*dir)[0,0]==delta:
            delta=min(2*delta,bound)
        else:
            delta=delta
        if rho>eta:
            x_1,x_2=x_1_prime,x_2_prime
        x.append([x_1,x_2])
    return x

print(trust_region(1.1,1.1,1,0.1,3,10))

def steepest_descet(x1,x2,k):
    x=[]
    for i in range(k):
        a=(100*pow(x1,2)+pow(x2,2))/(1000*pow(x1,2)+pow(x2,2))
        x1=x1-a*10*x1
        x2=x2-a*x2
        x.append([x1,x2])
    return x

print(steepest_descet(0.1,1,10))

def CG(x1,x2,k):
    A=np.asmatrix([[4,1],[1,3]])
    b=np.asmatrix([-1,-2]).T
    x=np.asmatrix([x1,x2]).T
    r=A*x+b
    p=-A*x-b
    for i in range(k):
        if r.all()!=0:
            a=(r.T*r)[0,0]/(p.T*A*p)[0,0]
            x=x+a*p
            r_prime=r+a*A*p
            beta=(r_prime.T*r_prime)[0,0]/(r.T*r)[0,0]
            p=-r_prime+beta*p
        break
    return x

#print(CG(2,1,4))