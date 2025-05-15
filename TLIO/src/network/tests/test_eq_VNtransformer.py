import pytest

import torch
from network.model_eq_VNtransformer import inner_dot_product

if __name__ == "__main__":
    x = torch.randn((2,200,35))
    y = torch.randn((2,200,35))
    res = inner_dot_product(x,y, dim=1)
    res.shape