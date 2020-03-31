import torch


def roll(t, n):
    return torch.cat((t[:, -n:], t[:, :-n]), dim=1)


def circular_convolution(w, s):
    w_p = torch.clone(w)
    for i in range(len(w)):
        for j in range(len(w[0])):
            w_p[i, j] = torch.mm(w, torch.t(roll(s, j)))[i]
    return w_p
