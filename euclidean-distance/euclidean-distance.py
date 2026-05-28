import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """

    if len(x) != len(y):
        raise ValueError("Vectors must have the same length")

    total = float(
        np.sqrt(
            np.sum(
                [(x[i] - y[i]) ** 2 for i in range(len(x))]
            )
        )
    )

    return total