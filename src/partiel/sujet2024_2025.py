import numpy as np
#                             I- Integration des EDO

# Ex_1 : 
def phi(t):
    return 0

def f(t, u, a):
    return np.array([u[1], u[2], a*np.sin(u[1] + u[0]**2)]) + phi(t)