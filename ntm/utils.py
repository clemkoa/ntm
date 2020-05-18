import torch
import torch.nn.functional as F


def roll(t, n):
    temp = t.flip(1)
    return torch.cat((temp[:, -(n+1):], temp[:, :-(n+1)]), dim=1)


def circular_convolution(w, s):
    temp_cat = torch.t(torch.cat([roll(s, i) for i in range(w.shape[1])]))
    return torch.mm(w, temp_cat)


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(1) == 3
    t = torch.cat([w[:, -1:], w, w[:, :1]], dim=1)
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(1, -1)
    return c
