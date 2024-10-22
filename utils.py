import os
import zipfile
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
from tqdm import tqdm
import gc


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    return None

def get_data_list(data_list):
    dlist = []
    split_data_list = list(data_list.split('_'))
    if('chengdu' in split_data_list):
        dlist.append('chengdu_m')
    if('metr' in split_data_list):
        dlist.append('metr-la')
    if('pems' in split_data_list):
        dlist.append('pems-bay')
    if('shenzhen' in split_data_list):
        dlist.append('shenzhen')
    return dlist

def unnorm(x ,means, stds):
    return x * stds[0] + means[0]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calc_metric(pred, y, stage = "train"):
    if(stage == "train"):
        MSE = torch.mean((pred - y)**2)
        RMSE = torch.sqrt(MSE)
        MAE = torch.mean(torch.abs(pred - y))
        MAPE = torch.mean(torch.abs((pred - y) / y))
    else:
        B, N, L = pred.shape
        pred = pred.reshape(-1, L)
        y = y.reshape(-1, L)
        MSE = torch.mean((pred - y)**2, dim = 0)
        RMSE = torch.sqrt(MSE)
        MAE = torch.mean(torch.abs(pred - y), dim = 0)
        MAPE = torch.mean(torch.abs((pred - y) / y), dim = 0)
    return MSE, RMSE, MAE, MAPE

def metric_func(pred, y, times):
    result = {}

    result['MSE'], result['RMSE'], result['MAE'], result['MAPE'] = np.zeros(times), np.zeros(times), np.zeros(times), np.zeros(times)

    # print("metric | pred shape:", pred.shape, " y shape:", y.shape)
    
    def cal_MAPE(pred, y):
        diff = np.abs(np.array(y) - np.array(pred))
        return np.mean(diff / y)

    for i in range(times):
        y_i = y[:,i,:]
        pred_i = pred[:,i,:]
        MSE = mean_squared_error(pred_i, y_i)
        RMSE = mean_squared_error(pred_i, y_i) ** 0.5
        MAE = mean_absolute_error(pred_i, y_i)
        MAPE = cal_MAPE(pred_i, y_i)
        result['MSE'][i] += MSE
        result['RMSE'][i] += RMSE
        result['MAE'][i] += MAE
        result['MAPE'][i] += MAPE
    return result

def result_print(result, info_name='Evaluate'):
    total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']
    print("========== {} results ==========".format(info_name))
    print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
    print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
    print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
    print("---------------------------------------")

    if info_name == 'Best':
        print("========== Best results ==========")
        print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
        print("---------------------------------------")


def load_data(dataset_name, stage):
    print("INFO: load {} data @ {} stage".format(dataset_name, stage))

    A = np.load("data/" + dataset_name + "/matrix.npy")
    A = get_normalized_adj(A)
    A = torch.from_numpy(A)
    X = np.load("data/" + dataset_name + "/dataset.npy")
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    # train: 70%, validation: 10%, test: 20%
    # source: 100%, target_1day: 288, target_3day: 288*3, target_1week: 288*7
    if stage == 'train':
        X = X[:, :, :int(X.shape[2]*0.7)]
    elif stage == 'validation':
        X = X[:, :, int(X.shape[2]*0.7):int(X.shape[2]*0.8)]
    elif stage == 'test':
        X = X[:, :, int(X.shape[2]*0.8):]
    elif stage == 'source':
        X = X
    elif stage == 'target_1day':
        X = X[:, :, :288]
    elif stage == 'target_3day':
        X = X[:, :, :288*3]
    elif stage == 'target_1week':
        X = X[:, :, :288*7]
    else:
        print("Error: unsupported data stage")

    print("INFO: A shape is {}, X shape is {}, means = {}, stds = {}".format(A.shape, X.shape, means, stds))

    return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds, inter_step, starting_time=0):
    
    print(X.shape)
    batch = (X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1 - starting_time) // inter_step
    print(f'{X.shape[2]} - ({num_timesteps_input} + {num_timesteps_output}) + 1 - {starting_time}) // {inter_step}')
    
    # print((batch, X.shape[0], X.shape[2], X.shape[1]))
    x = torch.zeros((batch+1, X.shape[0], num_timesteps_input, X.shape[1]))
    y = torch.zeros((batch+1, X.shape[0], num_timesteps_output))
    j = 0
    
    for i in range(starting_time, X.shape[2] - (num_timesteps_input + num_timesteps_output) + 1, inter_step):
        #print(i)
        x[j] = torch.from_numpy(X[:, :, i: i + num_timesteps_input].transpose((0, 2, 1))).float()
        y[j] = torch.from_numpy(X[:, 0, i + num_timesteps_input: i + num_timesteps_input + num_timesteps_output]*stds[0]+means[0]).float()
        j += 1

    return x, y

