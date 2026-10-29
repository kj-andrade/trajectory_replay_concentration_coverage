import numpy as np

def rk4_step(x,h,rhs,p):
    k1=rhs(x,p); k2=rhs(x+0.5*h*k1,p); k3=rhs(x+0.5*h*k2,p); k4=rhs(x+h*k3,p)
    return x+(h/6.)*(k1+2*k2+2*k3+k4)

def integrate(x0,n,h,rhs,p):
    out=np.empty((n+1,x0.shape[-1])); out[0]=x0; x=x0.copy()
    for i in range(n):
        x=rk4_step(x,h,rhs,p); out[i+1]=x
    return out

def flow_map(x,T,substeps,rhs,p):
    h=T/substeps; y=x.copy()
    for _ in range(substeps): y=rk4_step(y,h,rhs,p)
    return y
