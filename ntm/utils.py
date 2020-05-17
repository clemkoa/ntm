import torch


def roll(t, n):
    temp = t.flip(1)
    return torch.cat((temp[:, -(n+1):], temp[:, :-(n+1)]), dim=1)


def circular_convolution(w, s):
    temp_cat = torch.t(torch.cat([roll(s, i) for i in range(w.shape[1])]))
    return torch.mm(w, temp_cat)
