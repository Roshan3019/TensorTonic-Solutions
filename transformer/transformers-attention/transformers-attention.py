import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    scores = torch.matmul(Q, K.transpose(-2,-1))
    dim_k = K.shape[-1]
    scores = scores / math.sqrt(dim_k)
    attention_weight = F.softmax(scores, dim=-1)
    dot_product = torch.matmul(attention_weight,V)
    
    return dot_product