import torch.nn.functional as F
import torch
import torch.nn as nn
from torchmetrics.functional import pairwise_cosine_similarity

    
class Contrastive_loss_exp(nn.Module):
    def __init__(self, device, tau=0.1, contrastive_m=False):
        super(Contrastive_loss_exp, self).__init__()
        self.device = device
        self.tau = tau
        self.adj_m = contrastive_m

    def forward(self, x1, x2, A=None):
        n = x1.shape[1]
        if not self.adj_m:
            # A = torch.eye(n)
            A = torch.sign(torch.rand(n, n) - 0.1) + 1 + torch.eye(n)
            A[torch.where(A > 0)] = 1
        else:
            edges = torch.where(A != 0)
            A = torch.zeros((n, n))
            A[edges] = 1
            
        Y = (torch.ones(n, n) - A).to(f'cuda:{self.device}')
        A = torch.eye(n).to(f'cuda:{self.device}')
        
        if len(x1.shape) == 3:
            loss = torch.zeros(x1.shape[0])
            for i in range(x1.shape[0]):
                dist = torch.exp(pairwise_cosine_similarity(x1[i], x2[i]) / self.tau)
                loss[i] = -torch.sum(torch.log(torch.sum(A * dist, dim=1) / torch.sum(Y * dist, dim=1)))
            loss = torch.mean(loss)
        else:
            dist = torch.exp(pairwise_cosine_similarity(x1, x2) / self.tau)
            loss = -torch.sum(torch.log(torch.sum(A * dist, dim=1) / torch.sum(Y * dist, dim=1)))
        return loss
    