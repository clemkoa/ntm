import torch

from ..ntm.utils import circular_convolution, _convolve


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


def test_convolve():
    w = torch.tensor([0, 0, 1, 0, 0])
    s = torch.tensor([0, 1, 0])
    res = torch.tensor([0, 0, 1, 0, 0])
    assert torch.equal(_convolve(w, s), res)

    w = torch.tensor([0, 0, 1.0, 0, 0])
    s = torch.tensor([0.5, 0, 0.5])
    res = torch.tensor([0, 0.5, 0, 0.5, 0])
    assert torch.equal(_convolve(w, s), res)
