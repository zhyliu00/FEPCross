import torch
import torch.nn as nn
import sys
sys.path.append('../STmodels')
sys.path.append('../FEPCross_models')
from FEPCross import *
from gwn import *
sys.path.append('../../')
from utils import *
from contrastive_loss import *


def save_print(content, folder):
    with open(f'{folder}/log.out', 'a') as f:
        f.write(str(content))
        f.write('\n')
    print(content)


class st_fc(nn.Module):
    def __init__(self, device, dim, N):
        super(st_fc, self).__init__()
            
        self.STmodel = BatchA_patch_gwnet(out_dim=dim)
        
        self.cluster_query = nn.Embedding(4, 24)
        self.cluster_query_a = nn.Embedding(4, 24)
        self.cluster_query_phi = nn.Embedding(4, 24)
        
        self.fc = []
        self.fc = nn.Linear(384, 12).to(device)
        
        self.fc_2 = nn.Sequential(nn.Linear(12, 128))
        
        A = torch.randn(3 * N, 16)
        self.embd = nn.parameter.Parameter(A)
            
        self.gate_1 = nn.parameter.Parameter(torch.randn(128))
        self.gate_2 = nn.parameter.Parameter(torch.randn(128))
        self.gate_3 = nn.parameter.Parameter(torch.randn(128))
        
    def forward(self, x):
        return x

class FEPCross_combined(nn.Module):
    def __init__(self, encoders, device, model_args=None,args=None, encoders_trainable=False, alpha=1.0, generate_A=False, baseline=False, N=0):
        super(FEPCross_combined, self).__init__()
        self.device = device
        self.encoders = encoders
        self.generate_A = generate_A
        self.baseline = baseline
        self.N = N
        self.k = int(N/10)
        self.node_emb = torch.randn(N, 8)
        self.node_emb = nn.parameter.Parameter(self.node_emb)
        
        self.encoders_trainable = encoders_trainable
        
        self.aggregate = nn.Linear(24 * 128, 128)
        
        if not encoders_trainable:

            for param in self.encoders[0].parameters():
                param.requires_grad = False
        print("Train Encoder? {}".format(encoders_trainable))

        if baseline:
            dim = 12
        else:
            dim = 128
            
        self.STmodel = st_fc(self.device, dim, N).to(device)
        self.momentum_ratio = args.momentum_ratio
        self.momentum_A = None
        
        self.gate_A = nn.Parameter(torch.zeros(1))
        
    def generate_fake(self, x_f, x, x_a, A):
        self.encoders[0].mode = 'generate-fake'
        S = torch.squeeze(A[0])
        x_fake, x_a_fake, x_phi_fake = self.encoders[0]([x, x_f, x_a], S)
        B, N, _, _  = x_fake.shape
        x_fake = x_fake.reshape(B, N, -1)
        x_a_fake = x_a_fake.reshape(B, N, -1)
        x_phi_fake = x_phi_fake.reshape(B, N, -1)
        self.encoders[0].mode = 'Finetune'
        return x_fake, x_a_fake, x_phi_fake
        
    def forward(self, x_f, x, x_a, x_t_long, x_t, A, means, stds, train=False):
        
        S = torch.squeeze(A[0])
        en_f = self.encoders[0]([x, x_f, x_a], S)
        
        B, N, L, D = en_f.shape
        # print("en_f shape is : {}".format(en_f.shape))
        en_f = self.aggregate(en_f.reshape(B, N, L*D))
        en_f = torch.tanh(en_f)
        
        if(self.momentum_A == None):
            self.momentum_A = torch.squeeze(A[0]).unsqueeze(0).repeat(en_f.shape[0], 1, 1).detach()
            
        
        self.momentum_A = torch.mean(self.momentum_A, dim=0).unsqueeze(0).repeat(en_f.shape[0], 1, 1).detach()
        C = torch.softmax(torch.einsum("bij,bkj->bik",en_f,en_f), dim=-1)
        self.momentum_A = self.momentum_ratio * self.momentum_A + (1 - self.momentum_ratio) * C
        C = [self.momentum_A, self.momentum_A.transpose(1, 2)]
        
        
        en_t, _ = self.STmodel.STmodel(x_t, C)
        en_t = torch.squeeze(en_t)

        en_di = self.STmodel.fc_2(x_t.squeeze(-1))

        if self.baseline:
            o = en_t
        else:
            concat_input = torch.cat([en_t, en_f, en_di], dim=-1)
            o = self.STmodel.fc(concat_input)  
        
        o = o * stds + means
        return o
    