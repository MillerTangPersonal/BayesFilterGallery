import numpy as np

def skew(v): 
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], 
        [-v[1], v[0], 0]], dtype=float)
    
