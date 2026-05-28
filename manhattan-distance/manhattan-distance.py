import numpy as np

def manhattan_distance(x, y):
    """
    Compute the Manhattan (L1) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    total = int(np.sum([abs(x[i] - y[i]) for i in range(len(x))]))
    
        
    return total