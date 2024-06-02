import torch
import torch.nn as nn
import torch.nn.functional as F
        
     
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class BatchA_nconv(nn.Module):
    def __init__(self):
        super(BatchA_nconv,self).__init__()

    def forward(self,x, A):
        # here A : [B, N, N]
        # x : [B, dilation_channels, N, L2]
        
        # this step : [B, D, N, L] * [B, N, N] -> [B, D, N, L]
        x = torch.einsum('ncvl,nwv->ncwl',(x,A))
        return x.contiguous()    

class BatchA_gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(BatchA_gcn,self).__init__()
        self.nconv = BatchA_nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order
        self.support_len = support_len

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1): # original "for k in range(2, self.order + 1):"
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        # print("after gconv out dim = {}, x dim = {}".format(h.shape, x2.shape))
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class BatchA_patch_gwnet(nn.Module):
    def __init__(self, dropout=0.3, gcn_bool=True, in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2,supports_len=2):
        super(BatchA_patch_gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1

        # All supports are double transition
        self.supports_len = supports_len

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                # print("receptive_filed : {}, addtional_scope : {}, new_dilation : {}".format(receptive_field, additional_scope, new_dilation))
                if self.gcn_bool:
                    self.gconv.append(BatchA_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

        self.final_conv = nn.Linear(144,
                                    out_dim,)
        

        
        
    def forward(self, input, supports):
        if(not isinstance(supports, list)):
            supports = [supports]
        # input : [B, N, L, D]
        input = input.permute(0,3,1,2)
        # input : [B, D, N, L]
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        # print("x shape is : {}".format(x.shape))
        
        x = self.start_conv(x)
        # x : [B, residual_channels, N, L]
        skip = 0       

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            # residual = x
            
            ## here maybe the author write wrong code
            
            residual = x.clone()
            
            
            # dilated convolution
            # print("Conv2d input shape is ", residual.shape)
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            # print("Conv1d input shape is ", residual.shape)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # x : [B, dilation_channels, N, L2]
            
            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            # s : [B, dilation_channels, N, L2]
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            
            # skip = 0 thus skip = s
            # print(f's shape: {s.shape}')
            # print(f'x shape: {x.shape}')
            if self.gcn_bool and supports is not None:
                x = self.gconv[i](x,supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # print("x shape is : {}".format(x.shape))

        if(x.shape[-1]==1):
            x = (x.squeeze(-1)).permute(0,2,1)
        else:
            x = x.permute(0,2,1,3)
            x = torch.flatten(x, start_dim=2)
            x = self.final_conv(x)
                    
        return x, supports
