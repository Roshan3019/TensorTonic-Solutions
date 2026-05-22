import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    rows = len(A)
    cols = len(A[0])

    result = []
    for col in range(cols):
        new_row = []
        for row in range(rows):
            new_row.append(A[row][col])
        result.append(new_row)
    return np.array(result)
