import torch

from ..ntm.utils import circular_convolution


def test_circular_convolution():
    a = torch.tensor([[0, 0, 1, 0, 0]])
    b = torch.tensor([[1, 2, 3, 4, 5]])
    res = torch.tensor([[4, 5, 1, 2, 3]])
    assert torch.equal(circular_convolution(a, b), res)

    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[0, 0, 1, 0, 0]])
    res = torch.tensor([[4, 5, 1, 2, 3]])
    assert torch.equal(circular_convolution(a, b), res)

    a = torch.tensor([[1, 0, 1, 0, 0]])
    b = torch.tensor([[1, 2, 3, 4, 5]])
    res = torch.tensor([[5, 7, 4, 6, 8]])
    assert torch.equal(circular_convolution(a, b), res)

    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[1, 0, 1, 0, 0]])
    res = torch.tensor([[5, 7, 4, 6, 8]])
    assert torch.equal(circular_convolution(a, b), res)
