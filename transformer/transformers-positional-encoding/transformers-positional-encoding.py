import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    pe = np.zeros(
        (seq_length, d_model)
    )

    position = np.arange(seq_length).reshape(-1,1)

    div_term = np.exp(
        np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model)
    )

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe




    
    pos = 0
    index = (d_model / 2) - 1
    dim_list = []
    seq_dim = []
    for i in range(seq_length):
        pos = seq_length
        for j in range(index):
            exp_val = (2*index)/d_model
            internal_cal = pos/(10000^exp_val)
            PE_sin = np.sin(internal_cal)
            PE_cos = np.cos(internal_cal)
            dim_list.append(PE_sin, PE_cos)
        seq_dim.append(dim_list)
    