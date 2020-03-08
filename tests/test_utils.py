import pytest
import torch

from ..utils import circular_convolution

def test_circular_convolution():
    a = torch.tensor([[0, 0, 1, 0, 0]])
    b = torch.tensor([[1, 2, 3, 4, 5]])
    res = torch.tensor([[3, 2, 1, 5, 4]])
    assert torch.equal(circular_convolution(a, b), res)

    a = torch.tensor([[1, 0, 1, 0, 0]])
    b = torch.tensor([[1, 2, 3, 4, 5]])
    res = torch.tensor([[4, 7, 5, 8, 6]])
    assert torch.equal(circular_convolution(a, b), res)
