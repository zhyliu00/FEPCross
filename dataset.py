from enum import EnumMeta
from select import select
from typing import Union
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from utils import *
import random
from copy import deepcopy
import pandas as pd


random.seed(7)
class BBDefinedError(Exception):
    def __init__(self,ErrorInfo):
        super().__init__(self) 
        self.errorinfo=ErrorInfo
    def __str__(self):
        return self.errorinfo
    
def fetch_eigenvectors(A):
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    value, vector = np.linalg.eig(L)
    vector = vector.T
    index = np.argsort(np.absolute(value))
    value = np.absolute(value)[index]
    vector = np.absolute(vector)[index]
    return vector, value

def normalize(x):
    mean = torch.unsqueeze(torch.mean(x, dim=-1), dim=-1)
    std = torch.unsqueeze(torch.std(x, dim=-1), dim=-1)
    return (x - mean) / std

class traffic_dataset(Dataset):
    def __init__(self, data_args, task_args, data_list=None, stage='source', test_data='metr-la', add_target=True, target_days=2, frequency=False, starting_time=0, downSample=False):
        super(traffic_dataset, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.add_target = add_target
        self.test_data = test_data
        self.target_days = target_days
        self.frequency = frequency
        self.downSample = downSample
        self.predefined_data_list = data_list
        # if(self.stage == 'pretrain' or self.stage == 'cluster'):
        #     self.add_target = False
        self.starting_time = starting_time
        print('1: to load data')
        self.load_data(stage, test_data)
        
        print("[INFO] Dataset init finished!")


    # according to the stage, output x_list and y_list, both of them are dict
    def load_data(self, stage, test_data):
        self.A_list, self.edge_index_list = {}, {}
        self.edge_attr_list, self.node_feature_list = {}, {}
        self.x_list, self.y_list = {}, {}
        self.f_list, self.g_list = {}, {}
        self.fd_list, self.gd_list = {}, {}
        self.angle_list = {}
        self.means_list, self.stds_list = {}, {}
        self.batchnum_list = {}
        self.Laplacian_ev_list = {}
        self.node_embedding = {}
        
        data_keys = np.array(self.data_args['data_keys'])
        if(self.predefined_data_list != None):
            data_keys = self.predefined_data_list
            if self.add_target and self.test_data != 'None':
                data_keys += [self.test_data]
                
        if stage == 'source' or stage == 'pretrain' or self.stage == 'cluster' or self.stage == 'source_train':
            # self.data_list = np.delete(data_keys, np.where(data_keys == test_data))
            self.data_list = data_keys
            # self.data_list = np.array(['metr-la', 'chengdu_m'])
        elif stage == 'target' or stage == 'target_maml':
            self.data_list = np.array([test_data])
        elif stage == 'test':
            self.data_list = np.array([test_data])
        else:
            print("stage is : {}".format(stage))
            raise BBDefinedError('Error: Unsupported Stage')
        print("[INFO] {} dataset: {}".format(stage, self.data_list))

        print('')
        for dataset_name in self.data_list:
            print("dataset_name : {}".format(dataset_name))
            A = np.load(self.data_args[dataset_name]['adjacency_matrix_path'])
            vectors, values = fetch_eigenvectors(A)
            self.Laplacian_ev_list[dataset_name] = vectors
            edge_index, edge_attr, node_feature = self.get_attr_func(
            self.data_args[dataset_name]['adjacency_matrix_path']
            )

            self.A_list[dataset_name] = torch.from_numpy(get_normalized_adj(A))
            self.edge_index_list[dataset_name] = edge_index
            self.edge_attr_list[dataset_name] = edge_attr
            self.node_feature_list[dataset_name] = node_feature

            
            X = np.load(self.data_args[dataset_name]['dataset_path'])
            # [L, N, 4]
            # [:,:,0] : speed, [:,:,1] : some symbol of time?
            # (N, D, L)
            
            X = X.transpose((1, 2, 0))
            X = torch.tensor(X,dtype=torch.double)
            
            # [N, 2, L]
            X = torch.cat((X[:,0, :].unsqueeze(1), X[:,-1,:].unsqueeze(1)), dim = 1)
            
            # Interpolation. Chengdu and Shenzhen interpolated to 5min level.
            interp = False
            if self.stage == 'pretrain':
                if(dataset_name in ['chengdu_m','shenzhen']):
                    interp = True
            
            if(interp):
                interp_X = torch.nn.functional.interpolate(X, size = 2 * X.shape[-1] - 1,mode='linear',align_corners=True)
                # inter_speed.squeeze_(1)
                interp_X = torch.cat((interp_X[:,:,:1],interp_X),dim=-1)
                interp_X[:,1,0] = ((interp_X[:,1,1] - 1) + 2016 ) % 2016 # 2016 is the week slot
                X = interp_X

            X = X.numpy()
            
            # mean and std 
            X[:,0,:] = X[:,0,:].astype(np.float32)
            # print(X.shape)
            means = np.expand_dims(np.mean(X[:,0,:]),0)
            X[:,0,:] = X[:,0,:] - means.reshape(1, -1, 1)
            stds = np.expand_dims(np.std(X[:,0,:]),0)
            self.means_list[dataset_name], self.stds_list[dataset_name] = means, stds
            X[:,0,:] = X[:,0,:] / stds.reshape(1, -1, 1)
            
            
            # [N, 2, L] and 0 is normalized
            if stage == 'source' or stage == 'dann' or stage == 'pretrain' or stage == 'source_train':
                if(dataset_name == self.test_data):
                    X = X[:, :, :288 * self.target_days]
                else:
                    X = X

            # target, small sample to finetune, 288 = 24 * 12 is one day data.
            elif stage == 'target' or stage == 'target_maml':
                X = X[:, :, :288 * self.target_days]
                            
            # test, choose rest of data
            elif stage == 'test':
                X = X[:, :, 288 * self.target_days:]
            
            # X : [N, 2, L]
            
            if(self.stage == 'cluster'):
                self.x_list[dataset_name] = X
                self.y_list[dataset_name] = []
                continue
                
            # else:           
            his_num = self.task_args['his_num']
            pred_num = self.task_args['pred_num']
            
            
            if self.stage == 'pretrain':
                # gap 1 day
                inter_step = 12 * 5
            elif self.stage == 'source_train':
                inter_step = 12 * 23
            elif self.stage == 'target_maml':
                inter_step = 12 * 7
            else:
                inter_step = 12
            # x, y : [num_samples, num_vertices, L, D]
            # x, y : [B, N, L, D]
                
            x_inputs, y_outputs = generate_dataset(X, his_num, pred_num, means, stds, inter_step, starting_time=self.starting_time)
            
            rand_index = torch.randperm(x_inputs.shape[0])
            x_inputs = x_inputs[rand_index]
            y_outputs = y_outputs[rand_index]
            # x_data = deepcopy(x_inputs)
            # y_data = deepcopy(y_outputs)
            print(self.starting_time)
            
            if self.downSample and x_inputs.shape[0] != 0:
                downsample_rate = int(self.his_num / 1008)
                choose_index = downsample_rate * np.arange(1008)
                x_inputs = x_inputs[:, :, choose_index, :]
            
            print(x_inputs.shape)
            
            if self.frequency and x_inputs.shape[0] != 0:
                # frequency
                x_inputs_f = x_inputs.transpose(2, 3)[:, :, 0, :]
                x_fft = torch.fft.fft(x_inputs_f)
                x_inputs_f = torch.absolute(x_fft)
                x_angle = torch.angle(x_fft)
                node_shuffle = torch.randperm(x_inputs_f.shape[1])
                x_inputs_fd = x_inputs_f[:, node_shuffle, :]
                    
                index = torch.ones(x_inputs_f.shape) * torch.arange(x_inputs_f.shape[-1])
                index = torch.unsqueeze(index, -1)
                
                
                x_angle = torch.unsqueeze(x_angle, -1)
                x_inputs_f = torch.unsqueeze(x_inputs_f, -1) 
                x_inputs_f = torch.cat([x_inputs_f, index], dim=3)
                x_inputs_fd = torch.unsqueeze(x_inputs_fd, -1) 
                x_inputs_fd = torch.cat([x_inputs_fd, index], dim=3)
                x_angle = torch.cat([x_angle, index], dim=3)
                x_inputs_f = x_inputs_f[:, :, :, :x_inputs_f.shape[-1] // 2 + 1]
                x_inputs_fd = x_inputs_fd[:, :, :, :x_inputs_fd.shape[-1] // 2 + 1]
                    
                self.f_list[dataset_name] = x_inputs_f
                self.fd_list[dataset_name] = x_inputs_fd
                self.angle_list[dataset_name] = x_angle
                
            print('{} : x shape : {}, y shape : {}'.format(dataset_name, x_inputs.shape, y_outputs.shape))
            self.x_list[dataset_name] = x_inputs
            self.y_list[dataset_name] = y_outputs
        
        
        if(self.stage == 'pretrain' or self.stage == 'source_train'):
            self.pretrain_batchnum = 0
            batch_size = self.task_args['batch_size']
            for dataset_name in self.data_list:
                this_data_total_batches = int(self.x_list[dataset_name].shape[0] // batch_size)
                self.batchnum_list[dataset_name] = this_data_total_batches
                self.pretrain_batchnum += this_data_total_batches
                
            self.pretrain_which_data = torch.zeros((self.pretrain_batchnum))
            self.pretrain_which_pos = torch.zeros((self.pretrain_batchnum))
            cur = 0
            for idx, dataset_name in enumerate(self.data_list):
                self.pretrain_which_data[cur : cur + self.batchnum_list[dataset_name]] = int(idx)
                self.pretrain_which_pos[cur : cur + self.batchnum_list[dataset_name]] = torch.arange(cur, cur + self.batchnum_list[dataset_name]) - cur
                cur += self.batchnum_list[dataset_name]
            self.random_permutation =torch.randperm(self.pretrain_batchnum)
            
    def get_attr_func(self, matrix_path, edge_feature_matrix_path=None, node_feature_path=None):
        a, b = [], []
        edge_attr = []
        node_feature = None
        matrix = np.load(matrix_path)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if(matrix[i][j] > 0):
                    a.append(i)
                    b.append(j)
        edge = [a,b]
        edge_index = torch.tensor(edge, dtype=torch.long)

        return edge_index, edge_attr, node_feature
    
    def get_edge_feature(self, edge_index, x_data):
        pass
    
    
    def generate_fake(self, model, device, ratio, times=1):
        model.mode = 'generate-fake'
        select_dataset = self.data_list[0]
        f_list = [self.f_list[select_dataset]]
        a_list = [self.angle_list[select_dataset]]
        x_list = [self.x_list[select_dataset]]
        y_list = [self.y_list[select_dataset]]
        
        for i in range(times):
            f_list.append(self.f_list[select_dataset])
            a_list.append(self.angle_list[select_dataset])
            x_list.append(self.x_list[select_dataset])
            y_list.append(self.y_list[select_dataset])
            for j in range(6):
                A = self.A_list[select_dataset].float().to(device)
                x_f = self.f_list[select_dataset][4*j: 4*j+4].float().permute(0,1,3,2).to(device)
                x = self.x_list[select_dataset][4*j: 4*j+4].float().permute(0,1,3,2).to(device)
                x_a = self.angle_list[select_dataset][4*j: 4*j+4].float().permute(0,1,3,2).to(device)
                with torch.no_grad():
                    x_fake, x_a_fake, x_phi_fake = model([x, x_f, x_a], A)
                    # x_fake = model(x, A)
                B, N, _, _  = x_fake.shape
                x_fake = x_fake.cpu().reshape(B, N, -1)
                x_list[-1][4*j: 4*j+4, :, :, 0] = x_fake * ratio + x_list[-1][4*j: 4*j+4, :, :, 0] * (1 - ratio)
            

        self.x_list[select_dataset] = torch.cat(x_list, dim=0)
        self.angle_list[select_dataset] = torch.cat(a_list, dim=0)
        self.f_list[select_dataset] = torch.cat(f_list, dim=0)
        self.y_list[select_dataset] = torch.cat(y_list, dim=0)
        
        index = torch.randperm(self.x_list[select_dataset].shape[0])
        self.x_list[select_dataset] = self.x_list[select_dataset][index]
        self.angle_list[select_dataset] = self.angle_list[select_dataset][index]
        self.f_list[select_dataset] = self.f_list[select_dataset][index]
        self.y_list[select_dataset] = self.y_list[select_dataset][index]
        
        model.mode = 'Finetune'


    # index the dataset
    # used in finetune and testing
    def __getitem__(self, index):
        """
        : data.node_num record the node number of each batch
        : data.x shape is [batch_size, node_num, his_num, message_dim]
        : data.y shape is [batch_size, node_num, pred_num]
        : data.edge_index constructed for torch_geometric
        : data.edge_attr  constructed for torch_geometric
        : data.node_feature shape is [batch_size, node_num, node_dim]
        """

        if(self.stage == 'pretrain' or self.stage == 'source_train'):
            # need query *batch_size* continuous batches
            if self.frequency:
                idx = self.random_permutation[index]
                select_dataset = self.data_list[self.pretrain_which_data[idx].detach().cpu().numpy().astype(int)]
                pos = self.pretrain_which_pos[idx].detach().cpu().numpy().astype(int)
                batch_size = self.task_args['batch_size']
                # print('idx : {}, select_dataset : {}, pos : {}'.format(idx, select_dataset, pos))
                indices = torch.tensor(list(range(pos,pos+batch_size)))
                f_data = self.f_list[select_dataset][indices]
                fd_data = self.fd_list[select_dataset][indices]
                # g_data = self.g_list[select_dataset][indices]
                # gd_data = self.gd_list[select_dataset][indices]
                y_data = self.y_list[select_dataset][indices]
                x_data = self.x_list[select_dataset][indices]
                a_data = self.angle_list[select_dataset][indices]
            else:
                idx = self.random_permutation[index]
                select_dataset = self.data_list[self.pretrain_which_data[idx].detach().cpu().numpy().astype(int)]
                pos = self.pretrain_which_pos[idx].detach().cpu().numpy().astype(int)
                batch_size = self.task_args['batch_size']
                # print('idx : {}, select_dataset : {}, pos : {}'.format(idx, select_dataset, pos))
                indices = torch.tensor(list(range(pos,pos+batch_size)))
                x_data = self.x_list[select_dataset][indices]
                y_data = self.y_list[select_dataset][indices]

        # if 'source', randomly choose a city and random choose a batch
        elif (self.stage == 'source'):
            select_dataset = random.choice(self.data_list)
            batch_size = self.task_args['batch_size']
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
        
        # if 'target_maml', choose the first city and randomly choose a batch
        else:
            if(index == 0):
                self.random_permutation = torch.randperm(self.x_list[self.data_list[0]].shape[0] //  self.task_args['batch_size'])
            index = self.random_permutation[index]
            
            if self.frequency:
                select_dataset = self.data_list[0]
                batch_size = self.task_args['batch_size']
                # print("here batch_size is {}".format(batch_size))
                f_data = self.f_list[select_dataset][index * batch_size : index* batch_size + batch_size]
                fd_data = self.fd_list[select_dataset][index * batch_size : index* batch_size + batch_size]
                # g_data = self.g_list[select_dataset][index * batch_size : index* batch_size + batch_size]
                # gd_data = self.gd_list[select_dataset][index * batch_size : index* batch_size + batch_size]
                y_data = self.y_list[select_dataset][index * batch_size: index* batch_size + batch_size]
                x_data = self.x_list[select_dataset][index * batch_size : index* batch_size + batch_size]
                a_data = self.angle_list[select_dataset][index * batch_size : index* batch_size + batch_size]
                # print("here x_data shape is {}, y_data shape is {}".format(x_data.shape, y_data.shape))
            else:
                select_dataset = self.data_list[0]
                batch_size = self.task_args['batch_size']
                x_data = self.x_list[select_dataset][index * batch_size : index* batch_size + batch_size]
                y_data = self.y_list[select_dataset][index * batch_size: index* batch_size + batch_size]
            
        if self.frequency:
            f_data = f_data.float()
            fd_data = fd_data.float()
            y_data = y_data.float()
            x_data = x_data.float()
            a_data = a_data.float()
            data_i = Data(a=a_data, f=f_data, fd=fd_data, y=y_data, x=x_data, means=self.means_list[select_dataset],stds = self.stds_list[select_dataset])
        else:
            x_data = x_data.float()
            y_data = y_data.float()
            node_num = self.A_list[select_dataset].shape[0]
            data_i = Data(node_num=node_num, x=x_data, y=y_data, means=self.means_list[select_dataset],stds = self.stds_list[select_dataset])
        data_i.edge_index = self.edge_index_list[select_dataset]
        data_i.data_name = select_dataset
        A_wave = self.A_list[select_dataset]
        
        # x_data is [batch, n, HisStep, D], y_data is [batch, n, HisStep]
        # last, return data_i is a torch.geometric.data, contains x, y, edge index, which dataset
        # A_wave contains a adjacent matrix. Used to make reconstruction loss
        return data_i, A_wave
    
    
    # maml task, used in source training
    # each task is a graph. some batch of data on a graph
    def get_maml_task_batch(self, task_num):
        spt_task_data, qry_task_data = [], []
        spt_task_A_wave, qry_task_A_wave = [], []

        # first choose a random dataset
        select_dataset = random.choice(self.data_list)
        batch_size = self.task_args['batch_size']

        # equally distribute support set and qry set
        for i in range(task_num * 2):
            permutation = torch.randperm(self.x_list[select_dataset].shape[0])
            indices = permutation[0: batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
            node_num = self.A_list[select_dataset].shape[0]
            data_i = Data(node_num=node_num, x=x_data, y=y_data)
            data_i.edge_index = self.edge_index_list[select_dataset]
            # data_i.edge_attr = self.edge_attr_list[select_dataset]
            # data_i.node_feature = self.node_feature_list[select_dataset]
            data_i.data_name = select_dataset
            A_wave = self.A_list[select_dataset].float()

            # 
            
            if i % 2 == 0:
                spt_task_data.append(data_i.cuda())
                spt_task_A_wave.append(A_wave.cuda())
            else:
                qry_task_data.append(data_i.cuda())
                qry_task_A_wave.append(A_wave.cuda())

        return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave
    
    def __len__(self):
        if self.stage == 'source':
            print("[random permutation] length is decided by training epochs")
            return 100000000
        if self.stage == 'pretrain' or self.stage == 'source_train':
            # print("pretrain use datasets of {} cities".format(self.data_list))
            # [L, N, 2016, 2]
            return self.pretrain_batchnum
        if self.stage == 'target_maml' or self.stage == 'test' or self.stage == 'target':
            return int(self.x_list[self.data_list[0]].shape[0] //  self.task_args['batch_size'])
        else:
            data_length = self.x_list[self.data_list[0]].shape[0]
            return data_length



if __name__ == "__main__":
    
    
# # ----------------------- #
# #    test code(pretrain)
# # ----------------------- #
#     import yaml
#     with open('config.yaml') as f:
#             config = yaml.load(f)        

#     mydataset = traffic_dataset(config['data'], config['task']['mae'], stage='pretrain', test_data='metr-la',add_target=False)


#     train_batch_num = 10

#     for i in range(train_batch_num):
#         data, A_wave = mydataset[i]
#         print("node_num:{}, edge_index:{}, x:{}, A_wave:{}".format(data.node_num, data.edge_index.shape, data.x.shape, A_wave.shape))
        


#     # ----------------------- #
#     #    test code(source)
#     # ----------------------- #
#     import yaml
#     with open('config.yaml') as f:
#             config = yaml.load(f)        

#     mydataset = traffic_dataset(config['data'], config['task']['maml'], stage='source', test_data='metr-la')

#     train_batch_num = 10

#     for i in range(train_batch_num):
#         data, A_wave = mydataset[i]
#         print("node_num:{}, edge_index:{}, x:{}, y:{}, A_wave:{}".format(data.node_num, data.edge_index.shape, data.x.shape, data.y.shape, A_wave.shape))
    


# ----------------------- #
#    test code(test)
# ----------------------- #

    import yaml

    with open('config.yaml') as f:
            config = yaml.load(f)   
    data_list = "chengdu_shenzhen_metr"
    test_dataset = 'pems-bay'     
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    finetune_dataset = traffic_dataset(data_args, task_args['maml'], data_list, 'target_maml', test_data=test_dataset)
    test_dataset = traffic_dataset(data_args, task_args['maml'], data_list, 'test', test_data=test_dataset)

    print("length of dataset is", len(finetune_dataset))
    print("length of dataset is", len(test_dataset))

    print('finetune dataset')
    for idx in range(len(finetune_dataset)):
        data, A_wave = finetune_dataset[idx]
        print(idx, data, A_wave)
        print("node_num is {}, x_data shape is {}, y_data shape is {}".format(data.node_num, data.x.shape, data.y.shape))
        print("A_wave shape is", A_wave.shape)
    
    print('test_dataset')
    
    for idx in range(len(test_dataset)):
        data, A_wave = test_dataset[idx]
        print(idx, data, A_wave)
        print("node_num is {}, x_data shape is {}, y_data shape is {}".format(data.node_num, data.x.shape, data.y.shape))
        print("A_wave shape is", A_wave.shape)