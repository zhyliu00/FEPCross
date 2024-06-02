import torch
import torch.nn as nn
import gc
from FEPCross_models.FEPCross_layers import TransformerLayers, TransformerLayers_vanilla
from FEPCross_models.mask import MaskGenerator
from FEPCross_models.patch import Patch
from FEPCross_models.positional_encoding import PositionalEncoding

def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index

class FEPCross(nn.Module):
    def __init__(self, model_cfg, device, mode='Pretrain'):
        super().__init__()
        self.device = device
        patch_size, in_channel, out_channel, dropout, mask_size, mask_ratio, L = model_cfg['patch_size'], model_cfg['in_channel'], model_cfg['out_channel'], model_cfg['dropout'], model_cfg['mask_size'], model_cfg['mask_ratio'], model_cfg['L']
        self.patch_size = patch_size
        self.seleted_feature = 0
        self.mode = mode
        self.Gconv = model_cfg['Gconv']
        self.contrastive = model_cfg['Contrastive']
        self.patch = Patch(patch_size, in_channel, out_channel, spectral=False)
        self.pe = PositionalEncoding(out_channel, dropout=dropout)
        self.mask  = MaskGenerator(mask_size, mask_ratio)
        self.embed_out = nn.Linear(128*24, 32)
        
        L = 2
        
        self.encoder = TransformerLayers_vanilla(out_channel, L)
        self.encoder_a = TransformerLayers_vanilla(out_channel, L)
        self.encoder_phi = TransformerLayers_vanilla(out_channel, L)
        
        self.decoder = TransformerLayers(out_channel, L, Gconv=self.Gconv, device=device)
        self.encoder_2_decoder = nn.Linear(out_channel, out_channel)
        self.encoder_2_decoder_a = nn.Linear(out_channel, out_channel)
        self.encoder_2_decoder_phi = nn.Linear(out_channel, out_channel)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, out_channel))
        self.mask_token_a = nn.Parameter(torch.zeros(1, 1, 1, out_channel))
        self.mask_token_phi = nn.Parameter(torch.zeros(1, 1, 1, out_channel))

        nn.init.uniform_(self.mask_token, -0.02, 0.02)
        nn.init.uniform_(self.mask_token_a, -0.02, 0.02)
        nn.init.uniform_(self.mask_token_phi, -0.02, 0.02)
        
        self.output_layer = nn.Linear(out_channel, patch_size)
        self.output_layer_a = nn.Linear(out_channel, patch_size)
        self.output_layer_phi = nn.Linear(out_channel, patch_size)
        
    def _forward_pretrain(self, input, input_a, input_phi, A=None):
        """feed forward of the TSFormer in the pre-training stage.

        Args:
            input (torch.Tensor): very long-term historical time series with shape B, N, 2, L * P.
                                The first dimension is speed. The second dimension is position.
nn
        Returns:
            torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N]
            torch.Tensor: the groundtruth of the masked tokens. Shape [B, L * P * r, N]
            dict: data for plotting.
        """
        # input : [B, N, 2, L]
        B, N, C, L = input.shape
        position = input[:,:,1,:].unsqueeze(2)
        pos_indices = torch.arange(0, L, self.patch_size)
        position = position[:,:,:,pos_indices]
        # position : [B, N, 1, L/P]
        position = position // 12
        
        show_mid = False
        if(show_mid):
            torch.cuda.empty_cache()
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            
            for obj in gc.get_objects():
                try:    
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
            print('------------------------')
        
        # B, N, 1, L
        input = input[:,:,0,:].unsqueeze(2)
        input_a = input_a[:,:,0,:].unsqueeze(2)
        input_phi = input_phi[:,:,0,:].unsqueeze(2)

        # get patches and exec input embedding
        patches = self.patch(input)             # B, N, d, L/P
        patches = patches.transpose(-1, -2)     # B, N, L/P, d
        patches_a = self.patch(input_a)             # B, N, d, L/P
        patches_a = patches_a.transpose(-1, -2)     # B, N, L/P, d
        patches_phi = self.patch(input_phi)             # B, N, d, L/P
        patches_phi = patches_phi.transpose(-1, -2)     # B, N, L/P, d
        
        # positional embedding
        # patches : [B, N, L/P, d]. position : [B, N, 1, L/P]
        patches = self.pe(patches, position.long())
        
        if(show_mid):
            torch.cuda.empty_cache()
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                    
            for obj in gc.get_objects():
                try:    
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
            print('------------------------')
        
        # mask tokens
        # both 1D vector contains the index
        # 25, 75
        unmasked_token_index, masked_token_index = self.mask(self.mode)

        encoder_input = patches[:, :, unmasked_token_index, :]
        encoder_input_a = patches_a[:, :, unmasked_token_index, :]
        encoder_input_phi = patches_phi[:, :, unmasked_token_index, :]        

        # encoder
        construct_A = False
        # if self.mode == 'Pretrain':
        #     construct_A = True
        H = self.encoder(encoder_input)
        H_a = self.encoder_a(encoder_input_a)
        H_phi = self.encoder_phi(encoder_input_phi)
        
        # H, H_a, H_phi = self.encoder(encoder_input, encoder_input_a, encoder_input_phi, A, construct_A=construct_A)         # B, N, L/P*(1-r), d
        # encoder to decoder
        H = self.encoder_2_decoder(H)           # B, N, L/P*(1-r), d
        H_a = self.encoder_2_decoder_a(H_a)
        H_phi = self.encoder_2_decoder_phi(H_phi)
        
        # decoder
        # H_unmasked = self.pe(H, index=unmasked_token_index)
        
        if(show_mid):
            torch.cuda.empty_cache()
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                    
            for obj in gc.get_objects():
                try:    
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
            print('------------------------')
        
        # arg1 : [B, N, len(mti), d].  arg2 : [B, N, 1, L/P]
        masked_token_index_inpe = torch.tensor(masked_token_index)
        # position : [B, N, 1, len(mti)]
        indices = masked_token_index_inpe.expand(B, N, 1, len(masked_token_index))
        # pe input : patches : [B, N, len(mti), d]. position : [B, N, 1, len(mti)]
        H_masked   = self.pe(self.mask_token.expand(B, N, len(masked_token_index_inpe), H.shape[-1]), index=indices.long())
        H_masked_a   = self.pe(self.mask_token_a.expand(B, N, len(masked_token_index_inpe), H.shape[-1]), index=indices.long())
        H_masked_phi   = self.pe(self.mask_token_phi.expand(B, N, len(masked_token_index_inpe), H.shape[-1]), index=indices.long())
        
        
        ############
        # B, N, L/P, d
        H_full = torch.cat([H, H_masked], dim=-2)  
        H_full_a = torch.cat([H_a, H_masked_a], dim=-2)
        H_full_phi = torch.cat([H_phi, H_masked_phi], dim=-2)
        # B, N, L/P, d

        H, H_a, H_phi = self.decoder(H_full, H_full_a, H_full_phi, A, construct_A)
        
        if self.contrastive and self.mode != 'generate-fake':
            B, N, lp, d = H.shape
            decoder_output = (H + H_a + H_phi) / 3
            decoder_output = decoder_output.reshape(B, N, lp*d)
            # rand_w = torch.randn(lp*d, 32).to(self.device)
            # decoder_output = decoder_output @ rand_w
            decoder_output = self.embed_out(decoder_output)


        if(show_mid):
            torch.cuda.empty_cache()
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                    
            for obj in gc.get_objects():
                try:    
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
            print('------------------------')
        
        # output layer
        # B, N, L/P, P
        out_full = self.output_layer(H)
        out_full_a = self.output_layer_a(H_a)
        out_full_phi = self.output_layer_phi(H_phi)

        # prepare loss
        B, N, _, _ = out_full.shape 
        # B, N, len(mask), P
        out_masked_tokens = out_full[:, :, len(unmasked_token_index):, :]
        # B, len(mask) * P, N
        out_masked_tokens = out_masked_tokens.view(B, N, -1).transpose(1, 2)
        
        out_masked_tokens_a = out_full_a[:, :, len(unmasked_token_index):, :]
        # B, len(mask) * P, N
        out_masked_tokens_a = out_masked_tokens_a.view(B, N, -1).transpose(1, 2)
        
        out_masked_tokens_phi = out_full_phi[:, :, len(unmasked_token_index):, :]
        # B, len(mask) * P, N
        out_masked_tokens_phi = out_masked_tokens_phi.view(B, N, -1).transpose(1, 2)
        
        if(show_mid):
            torch.cuda.empty_cache()
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                    
            for obj in gc.get_objects():
                try:    
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
            print('------------------------')

        # B, N, 1, L -> B, L, N, 1 -> B, L/P, N, 1, P -> B, L/P, N, P -> B, N, L/P, P
        label_full  = input.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.seleted_feature, :].transpose(1, 2)  # B, N, L/P, P
        # B, N, L/P * r, P
        label_masked_tokens  = label_full[:, :, masked_token_index, :].contiguous()
        # B, N, L/p * r * P -> B, L/p * r * P, N
        label_masked_tokens  = label_masked_tokens.view(B, N, -1).transpose(1, 2)
        
        label_full_a  = input_a.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.seleted_feature, :].transpose(1, 2)  # B, N, L/P, P
        # B, N, L/P * r, P
        label_masked_tokens_a  = label_full_a[:, :, masked_token_index, :].contiguous()
        # B, N, L/p * r * P -> B, L/p * r * P, N
        label_masked_tokens_a  = label_masked_tokens_a.view(B, N, -1).transpose(1, 2)
        
        label_full_phi  = input_phi.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.seleted_feature, :].transpose(1, 2)  # B, N, L/P, P
        # B, N, L/P * r, P
        label_masked_tokens_phi  = label_full_phi[:, :, masked_token_index, :].contiguous()
        # B, N, L/p * r * P -> B, L/p * r * P, N
        label_masked_tokens_phi  = label_masked_tokens_phi.view(B, N, -1).transpose(1, 2)

        # prepare plot
        ## note that the output_full and label_full are not aligned. The out_full is shuffled.
        
        # index is the position in L/P, value is the actual position in the H : [B, N, (L/P), d]
        unshuffled_index = unshuffle(unmasked_token_index + masked_token_index)     # therefore, we need to unshuffle the out_full for better plotting.
        
        # B, N, L/P, P
        out_full_unshuffled = out_full[:, :, unshuffled_index, :]
        out_full_unshuffled_a = out_full_a[:, :, unshuffled_index, :]
        out_full_unshuffled_phi = out_full_phi[:, :, unshuffled_index, :]
        plot_args = {}
        plot_args['out_full_unshuffled']    = out_full_unshuffled
        plot_args['label_full']             = label_full
        plot_args['unmasked_token_index']   = unmasked_token_index
        plot_args['masked_token_index']     = masked_token_index
        
        if(show_mid):
            torch.cuda.empty_cache()
            print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                    
            for obj in gc.get_objects():
                try:    
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
            print('------------------------')
        
        # return
        # torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N]
        # torch.Tensor: the groundtruth of the masked tokens. Shape [B, L * P * r, N]
        # dict: data for plotting.
        if self.contrastive:
            if self.mode == 'generate-fake':
                print(unmasked_token_index)
                print(masked_token_index)
                
                out_full_unshuffled[:, :, unmasked_token_index, :] = label_full[:, :, unmasked_token_index, :].contiguous()
                
                out_full_unshuffled_a[:, :, unmasked_token_index, :] = label_full_a[:, :, unmasked_token_index, :].contiguous()
                
                out_full_unshuffled_phi[:, :, unmasked_token_index, :] = label_full_phi[:, :, unmasked_token_index, :].contiguous()
                
                return out_full_unshuffled, out_full_unshuffled_a, out_full_unshuffled_phi
                
            return out_masked_tokens, label_masked_tokens, \
                out_masked_tokens_a, label_masked_tokens_a, \
                out_masked_tokens_phi, label_masked_tokens_phi, plot_args, decoder_output
        
        return out_masked_tokens, label_masked_tokens, \
                out_masked_tokens_a, label_masked_tokens_a, \
                out_masked_tokens_phi, label_masked_tokens_phi, plot_args

    def _forward_backend(self, input, input_a, input_phi, A=None):
        """the feed forward process in the forecasting stage.

        Args:
            input (torch.Tensor): very long-term historical time series with shape B, N, 1, L * P.

        Returns:
            torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, d].
        """
        # B, N, C, LP = input.shape
        # # get patches and exec input embedding
        # patches = self.patch(input)             # B, N, d, L
        # patches = patches.transpose(-1, -2)     # B, N, L, d
        # # positional embedding
        # patches = self.pe(patches)
        
        # encoder_input = patches          # no mask when running the backend.

        # # encoder
        # H = self.encoder(encoder_input)         # B, N, L, d
                # input : [B, N, 2, L]
        B, N, C, L = input.shape
        position = input[:,:,1,:].unsqueeze(2)
        pos_indices = torch.arange(0, L, self.patch_size)
        position = position[:,:,:,pos_indices]
        # position : [B, N, 1, L/P]
        position = position // 12
        position = position % 168 # 168 = 24 * 7
        
        # prefix tuning
        if(L > 288):
            position[:,:,:,0]=169
            position[:,:,:,1]=170
        
        # B, N, 1, L
        input = input[:,:,0,:].unsqueeze(2)
        input_a = input_a[:,:,0,:].unsqueeze(2)
        input_phi = input_phi[:,:,0,:].unsqueeze(2)
        # get patches and exec input embedding
        patches = self.patch(input).transpose(-1, -2)             # B, N, d, L/P
        patches_a = self.patch(input_a).transpose(-1, -2)             # B, N, d, L/P
        patches_phi = self.patch(input_phi).transpose(-1, -2)             # B, N, d, L/P
        
        
        # positional embedding
        # patches : [B, N, L/P, d]. position : [B, N, 1, L/P]
        patches = self.pe(patches, position.long())

        # encoder
        # H, H_a, H_phi = self.encoder(patches, patches_a, patches_phi, A) # B, N, L/P, d
        H = self.encoder(patches)
        H_a = self.encoder_a(patches_a)
        H_phi = self.encoder_phi(patches_phi)
        
        # print("Here1 H shape is : {}".format(H.shape))
        # print("1 : H shape is : {}".format(H.shape))
        
        H = self.encoder_2_decoder(H)           
        H_a = self.encoder_2_decoder_a(H_a)
        H_phi = self.encoder_2_decoder_phi(H_phi)
        # print("2 : H shape is : {}".format(H.shape))
        construct_A = False
        
        # print("Here2 H shape is : {}".format(H.shape))
        H, H_a, H_phi = self.decoder(H, H_a, H_phi, A, construct_A)
        H = (H + H_a + H_phi) / 3
        # print("3 H shape is : {}".format(H.shape))
        return H

    def forward(self, input_data, A=None):
        """feed forward of the TSFormer.
        TSFormer has two modes: the pre-training mode and the forecasting model, which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            input_data (torch.Tensor): very long-term historical time series with shape B, N, 1, L * P.
        
        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N]
                torch.Tensor: the groundtruth of the masked tokens. Shape [B, L * P * r, N]
                dict: data for plotting.
            forecasting: 
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, d].
        """
        if self.mode == 'Pretrain' or self.mode == 'generate-fake':
            return self._forward_pretrain(input_data[0], input_data[1], input_data[2], A)
        else:
            return self._forward_backend(input_data[0], input_data[1], input_data[2], A)
    def back(self, pattern):
        # pattern : [1, 1, K, D]
        # pattern = pattern.unsqueeze(0).unsqueeze(0)
        pattern = pattern.unsqueeze(1).unsqueeze(1)
        mid = self.decoder(pattern)
        res = self.output_layer(mid)
        return res

