import math
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch
from torch_geometric.nn import SGConv
import numpy as np



class Spatial_layer(nn.Module):
    def __init__(self, hidden_dim, length):
        super().__init__()
        self.length = length
        self.hidden_dim = hidden_dim
        
        # (B, N, L, D) -> (B, N, L*D) -> (B, N, L/3*D)
        self.first_linear = nn.Linear(length * hidden_dim, int(length/4) * int(hidden_dim/2))
        # self.W = torch.randn(int(length/4) * int(hidden_dim/2),  int(length/4) * int(hidden_dim/2))
        self.W = nn.Linear(int(length/4) * int(hidden_dim/2),  length * hidden_dim)
        # self.second_linear = nn.Linear(int(length/4) * int(hidden_dim/2), length * hidden_dim)
        self.activation = nn.Tanh()
    def forward(self, X, A):
        # input: (B, N, L, D)
        B, N, L, D = X.shape
        flg = 0
        # print("Here L is {}, self.length is {}".format(L, self.length))
        if(L > self.length):
            prefix = X[:, :, :L-self.length, :]
            X_1 = X[:, :, -self.length:, :]
            L = self.length
            flg = 1
        else:
            X_1 = X
        X_1 = X_1.reshape(B, N, L*D)
        X_1 = self.first_linear(X_1)
        X_1 = torch.einsum('nc,bcd->bnd', A, X_1)
        X_1 = self.activation(self.W(X_1))
        # X = self.activation(self.second_linear(X))
        X_1 = X_1.reshape(B, N, L, D)
        if(flg):
            X_1 = torch.cat([prefix, X_1], dim = 2)
            
        return X_1

        
class OnelayerTST(nn.Module):
    def __init__(self, hidden_dim, nlayers, device, num_heads=4, dropout=0.1, Gconv=False):
        super().__init__()
        self.Gconv = Gconv
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
        self.transformer_encoder_total = TransformerEncoder(encoder_layers, nlayers)
        self.device = device
        self.cnt = 110
        
        encoder_layers_t = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
        self.transformer_encoder_t = TransformerEncoder(encoder_layers_t, nlayers)
        encoder_layers_a = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
        self.transformer_encoder_a = TransformerEncoder(encoder_layers_a, nlayers)
        encoder_layers_phi = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
        self.transformer_encoder_phi = TransformerEncoder(encoder_layers_phi, nlayers)
        
        decoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)        
        self.transformer_decoder_total = TransformerEncoder(decoder_layers, 1)
        
        decoder_layers_t = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
        self.transformer_decoder_t = TransformerEncoder(decoder_layers_t, nlayers)
        decoder_layers_a = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
        self.transformer_decoder_a = TransformerEncoder(decoder_layers_a, nlayers)
        decoder_layers_phi = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
        self.transformer_decoder_phi = TransformerEncoder(decoder_layers_phi, nlayers)
            
        self.Spaital_layer = Spatial_layer(hidden_dim, 24)
        self.Spaital_layer_a = Spatial_layer(hidden_dim, 24)
        self.Spaital_layer_phi = Spatial_layer(hidden_dim, 24)
        
        
    def forward(self, src, src_a, src_phi, A=None, construct_A=False):
        B, N, L, D = src.shape
        
        src = src * math.sqrt(self.d_model)
        src = src.view(B*N, L, D).transpose(0, 1)
        src_a = src_a.view(B*N, L, D).transpose(0, 1)
        src_phi = src_phi.view(B*N, L, D).transpose(0, 1)
        
        # [3L, BN, D]
        input_concat = torch.cat([src, src_a, src_phi], dim = 0)
        
        with torch.no_grad():
            x_tmp = torch.unsqueeze(input_concat[:, 0, :], dim=1)
            layer = self.transformer_encoder_total.layers[0]
            _, x_tmp = layer.self_attn(x_tmp, x_tmp, x_tmp, need_weights=True)
            x_tmp = torch.squeeze(x_tmp).detach().cpu().numpy()
            np.save(f'/FSL_spectral/base/data/cross-relation-{self.cnt}.npy', x_tmp)
            self.cnt += 1
        
        output_concat = self.transformer_encoder_total(input_concat, mask = None).transpose(0, 1).view(B, N, L*3, D)
        
        output = output_concat[:, :, :L, :]
        output_a = output_concat[:, :, L:2*L, :]
        output_phi = output_concat[:, :, 2*L:, :]
        
        # output = self.transformer_encoder_t(src).transpose(0, 1).view(B, N, L, D)
        # output_a = self.transformer_encoder_a(src_a).transpose(0, 1).view(B, N, L, D)
        # output_phi = self.transformer_encoder_phi(src_phi).transpose(0, 1).view(B, N, L, D)
        
        output = self.Spaital_layer(output, A)
        output_a = self.Spaital_layer_a(output_a, A)
        output_phi = self.Spaital_layer_phi(output_phi, A)
        
        # output = self.transformer_decoder_t(output.view(B*N, L, D).transpose(0, 1)).transpose(0, 1).view(B, N, L, D)
        # output_a = self.transformer_decoder_a(output_a.view(B*N, L, D).transpose(0, 1)).transpose(0, 1).view(B, N, L, D)
        # output_phi = self.transformer_decoder_phi(output_phi.view(B*N, L, D).transpose(0, 1)).transpose(0, 1).view(B, N, L, D)
        
        
        # three [B, N, L, D]
        B, N, L, D = output.shape
        input_concat = torch.cat([output, output_a, output_phi], dim = 2).reshape(B*N, L*3, D).transpose(0, 1)
        # [3L, BN, D]
        
        
        
        output_concat = self.transformer_decoder_total(input_concat, mask = None).transpose(0, 1).view(B, N, L*3, D)
        
        output = output_concat[:, :, :L, :]
        output_a = output_concat[:, :, L:2*L, :]
        output_phi = output_concat[:, :, 2*L:, :]
        
        # output, output_a, output_phi = src, src_a, src_phi

        return output, output_a, output_phi

class TransformerLayers(nn.Module):
    def __init__(self, hidden_dim, nlayers, device, num_heads=4, dropout=0.1, Gconv=False):
        super().__init__()
        self.Gconv = Gconv
        self.d_model = hidden_dim
        self.first=OnelayerTST(hidden_dim, nlayers, device, num_heads, dropout, Gconv)
        # self.second=OnelayerTST(hidden_dim, nlayers, device, num_heads, dropout, Gconv)
        self.device = device
        hidden_dim = int(hidden_dim)
        print('Using Gconv in Transformer.')
        city_1 = torch.randn(524, 8)
        city_2 = torch.randn(207, 8)
        city_3 = torch.randn(325, 8)
        city_4 = torch.randn(627, 8)

        self.emb1 = nn.parameter.Parameter(city_1).to(self.device)
        self.emb2 = nn.parameter.Parameter(city_2).to(self.device)
        self.emb3 = nn.parameter.Parameter(city_3).to(self.device)
        self.emb4 = nn.parameter.Parameter(city_4).to(self.device)
    def forward(self, src, src_a, src_phi, A=None, construct_A=False):
        B, N, L, D = src.shape
        # if self.Gconv:
        
        #     # if construct_A:
        #     if N == 524:
        #         A_hat = self.emb1 @ self.emb1.T
        #     elif N == 207:
        #         A_hat = self.emb2 @ self.emb2.T
        #     elif N == 325:
        #         A_hat = self.emb3 @ self.emb3.T
        #     else:
        #         A_hat = self.emb4 @ self.emb4.T
            
        #     S = torch.softmax((A_hat + A_hat.T) / 2, 0)
        #     # S = torch.zeros((3 * N, 3 * N)).to(self.device)
        #     # S[:N, :N] = A
        #     # S[N :2 * N, N :2 * N] = A
        #     # S[2 * N :3 * N, 2 * N: 3 * N] = A
        #     degree_m = torch.diag(torch.sum(S, dim=0)**(-0.5)).to(self.device)
        #     M = (torch.eye(N).to(self.device) + degree_m @ S @ degree_m).float()
        M = A
        output, output_a, output_phi=self.first(src, src_a, src_phi, M, construct_A)
        # output, output_a, output_phi=self.second(output, output_a, output_phi, M, construct_A)
        return output, output_a, output_phi




class TransformerLayers_vanilla(nn.Module):
    def __init__(self, hidden_dim, nlayers, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

    def forward(self, src):
        B, N, L, D = src.shape
        src = src * math.sqrt(self.d_model)
        src = src.view(B*N, L, D)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, mask=None)
        output = output.transpose(0, 1).view(B, N, L, D)
        
        # return [B, N, L, D]
        
        return output