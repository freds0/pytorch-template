import torch.nn.functional as F

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    return F.cross_entropy(output, target)
