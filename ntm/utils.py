import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def roll(t, n):
    temp = t.flip(1)
    return torch.cat((temp[:, -(n+1):], temp[:, :-(n+1)]), dim=1)


def circular_convolution(w, s):
    temp_cat = torch.t(torch.cat([roll(s, i) for i in range(w.shape[1])]))
    return torch.mm(w, temp_cat)


def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    t = torch.cat([w[-1:], w, w[:1]], dim=0)
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c


def plot_copy_results(target, y, vector_length):
    plt.set_cmap('jet')
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.set_ylabel("target", rotation=0, labelpad=20)
    ax1.imshow(torch.t(target.view(-1, vector_length)))
    ax1.tick_params(axis="both", which="both", length=0)
    ax2 = fig.add_subplot(212)
    ax2.set_ylabel("output", rotation=0, labelpad=20)
    ax2.imshow(torch.t(y.clone().data.view(-1, vector_length)))
    ax2.tick_params(axis="both", which="both", length=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.show()
