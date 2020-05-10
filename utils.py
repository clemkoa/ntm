import torch


def roll(t, n):
    return torch.cat((t[:, -n:], t[:, :-n]), dim=1)


def circular_convolution(w, s):
    temp_cat = torch.t(torch.cat([roll(s, i) for i in range(w.shape[1])]))
    return torch.mm(w, temp_cat)
