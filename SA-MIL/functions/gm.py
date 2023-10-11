import torch
def gm(x, r):
    # return torch.pow(torch.pow(x.reshape(x.shape[0], -1), r).mean(dim=1).unsqueeze(1), 1/r)
    return torch.pow(torch.pow(x, r).mean(-1).mean(-1), 1/r)