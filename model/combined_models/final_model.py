import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import time
from copy import deepcopy
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
import math
from tqdm import trange
from torch.utils.data import DataLoader
from combined_model import *

def save_print(content, folder):
    with open(f'{folder}/log.out', 'a') as f:
        f.write(str(content))
        f.write('\n')
    print(content)
    
    
class FEPCross_wrap(nn.Module):
    """
    Reptile-based Few-shot learning architecture for STGNN
    """
    def __init__(self, encoders, gwn, device, folder, model_args=None,args=None,update_lr=3e-4, meta_lr=3e-4, update_step=3, task_num=1, epochs=200, en_trainable=False, baseline=False, node_num=0):
        super(FEPCross_wrap, self).__init__()

        self.update_lr = update_lr
        self.meta_lr = meta_lr
        print("[INFO] in Fre_Rep, meta_lr is {}".format(self.meta_lr))
        self.update_step = update_step
        # update_step_test is not used. It is replaced by target_epochs in main.
        self.task_num = task_num
        self.device = device
        self.current_epoch = 0
        self.folder = folder
        self.epochs = epochs
        
        self.model = FEPCross_combined(encoders, device, model_args = model_args,args=args, encoders_trainable=en_trainable, generate_A=True, baseline=baseline, N=node_num).to(self.device)

        model_params = count_parameters(self.model)
        save_print(f"model params: {model_params}", self.folder)

        self.meta_optim = optim.AdamW(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-2)
        self.loss_criterion = nn.MSELoss(reduction='mean')

    def finetune(self, time_dataset, epochs=60):
        best_loss = 9999999999999.0
        best_model = None
        optim = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, gamma=0.2, step_size=450)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, gamma=0.5, milestones=[2, 5, 10, 40, 80, 200])
        criterion = nn.MSELoss().to(self.device)

        for i in range(epochs):
            t1 = time.time()
            length = time_dataset.__len__()
            save_print('----------------------', self.folder)
            total_loss = []
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            print("[INFO] in fine-tune, dataset length is {}".format(length))
            for j in range(length):
                data_i_3, A = time_dataset[j]
                A = A.unsqueeze(0).float().to(self.device)
                x_t = data_i_3.x.to(self.device)
                
                x_f = data_i_3.f.permute(0,1,3,2).to(self.device)
                x_t_long = torch.squeeze(x_t[:, :, :, 0:1])
                x_t_long = torch.unsqueeze(x_t_long, dim=1)
                raw_x = x_t[:, :, -12:, 0:1]
                x_t = data_i_3.x.permute(0,1,3,2).to(self.device)
                x_a = data_i_3.a.permute(0,1,3,2).to(self.device)
                
                
                y = data_i_3.y.to(self.device)
                means = data_i_3.means[0]
                stds = data_i_3.stds[0]
                pred = self.model(x_f, x_t, x_a, x_t_long, raw_x, [A, A.transpose(1, 2)], means=means, stds=stds, train=True)
                # print("here pred shape is : {}".format(pred.shape))
                optim.zero_grad()
                loss = criterion(pred, y)
                # print(loss)
                loss.backward()
                optim.step()
                
                MSE, RMSE, MAE, MAPE = calc_metric(pred, y)
                total_mse.append(MSE.cpu().detach().numpy())
                total_rmse.append(RMSE.cpu().detach().numpy())
                total_mae.append(MAE.cpu().detach().numpy())
                total_mape.append(MAPE.cpu().detach().numpy())
                total_loss.append(loss.item())
            
            save_print(f'Epoch: {i}', self.folder)
            save_print('in training Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, normed MSE : {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape),np.mean(total_loss)), self.folder)
            scheduler.step()
            
            mae_loss = np.mean(total_mae)
            if mae_loss < best_loss:
                torch.save(self.model.state_dict(), f'{self.folder}/finetuned_bestmodel.pt')
                best_loss = mae_loss
                save_print('Best model Saved.', self.folder)
            save_print('this epoch costs {:.5}s'.format(time.time() - t1), self.folder)
            
    def test(self, time_dataset):
        save_print('----------------------', self.folder)
        save_print('Start testing', self.folder)
        length = time_dataset.__len__()

        print("[INFO] in test, dataset length is {}".format(length))
        total_mae_horizon = []
        total_mse_horizon = []
        total_rmse_horizon = []
        total_mape_horizon = []
        
        with torch.no_grad():
            for j in range(length):

                data_i_3, A = time_dataset[j]
                A = A.unsqueeze(0).float().to(self.device)
                x_t = data_i_3.x.to(self.device)
                x_f = data_i_3.f.permute(0,1,3,2).to(self.device)
                x_t_long = torch.squeeze(x_t[:, :, :, 0:1])
                x_t_long = torch.unsqueeze(x_t_long, dim=1)
                raw_x = x_t[:, :, -12:, 0:1]
                x_t = data_i_3.x.permute(0,1,3,2).to(self.device)
                x_a = data_i_3.a.permute(0,1,3,2).to(self.device)
                
                
                y = data_i_3.y.to(self.device)
                
                means = data_i_3.means[0]
                stds = data_i_3.stds[0]
                pred = self.model(x_f, x_t, x_a, x_t_long, raw_x, [A, A.transpose(1, 2)], means=means, stds=stds)
                # print("here pred shape is : {}".format(pred.shape))
                MSE, RMSE, MAE, MAPE = calc_metric(pred, y, stage='test')
                total_mse_horizon.append(MSE.cpu().detach().numpy())
                total_rmse_horizon.append(RMSE.cpu().detach().numpy())
                total_mae_horizon.append(MAE.cpu().detach().numpy())
                total_mape_horizon.append(MAPE.cpu().detach().numpy())
                
        for i in range(12):
            total_mae = []
            total_mse = []
            total_rmse = []
            total_mape = []
            for j in range(len(total_mse_horizon)):
                
                total_mse.append(total_mse_horizon[j][i])
                total_rmse.append(total_rmse_horizon[j][i])
                total_mae.append(total_mae_horizon[j][i])
                total_mape.append(total_mape_horizon[j][i])
                
            save_print('Horizon {} : Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(i,np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)), self.folder)
            
                            