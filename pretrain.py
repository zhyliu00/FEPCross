import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from dataset import traffic_dataset
from utils import *
import argparse
import yaml
import time
from pathlib import Path
import sys
from tqdm import trange
sys.path.append('./model')
sys.path.append('./model/FEPCross_models')
sys.path.append('./model/STmodels')
sys.path.append('./model/combined_models')
from FEPCross import *
from contrastive_loss import *


parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='./configs/config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='chengdu_m', type=str)
parser.add_argument('--data_list', default='shenzhen_pems_metr_chengdu',type=str)
parser.add_argument('--gpu', default=0, type = int)
parser.add_argument('--seed', default=7, type = int)
args = parser.parse_args()

contrastive_loss = Contrastive_loss_exp(device=args.gpu, contrastive_m=False)

seed=args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float32)
# print(time.strftime('%Y-%m-%d %H:%M:%S'), "meta_dim = ", args.meta_dim,"target_days = ", args.target_days)



def train_batch(start,end,model,source_dataset,loss_fn,opt):
    total_loss = []
    total_mae = []
    total_mse = []
    total_rmse = []
    total_mape = []
    contr_loss = []
    model.train()
    total_loss_1, total_loss_2, total_loss_3 = 0, 0, 0
    for idx in trange(start, end, ncols=80):
        data_i, A_wave = source_dataset[idx]
        # [B, N, L, 2]
        x, a, phi, means, stds = data_i.x, data_i.f, data_i.a, data_i.means, data_i.stds
        a_aug = data_i.fd
        
        A = torch.squeeze(A_wave).to(args.device).to(torch.float32)
        
        B, N, L, _ = x.shape
        # remember that the input of TSFormer is [B, N, 2, L]
        
        x = x.permute(0,1,3,2).to(args.device)
        a = a.permute(0,1,3,2).to(args.device)
        a_aug = a_aug.permute(0,1,3,2).to(args.device)
        phi = phi.permute(0,1,3,2).to(args.device)
        
        o1, l1, o2, l2, o3, l3, plot_args, en = model([x, a, phi], A)
        
        _, _, _, _, _, _, _, en_aug = model([x, a_aug, phi], A)
        
        # only the masked patch is loss target 
        loss_0 = contrastive_loss(en, en_aug, A)
        loss_1 = loss_fn(o1, l1)
        loss_2 = loss_fn(o2, l2)
        loss_3 = loss_fn(o3, l3)
        loss = loss_1/loss_1.item() + loss_2/loss_2.item() + loss_3/loss_3.item() + loss_0/loss_0.item()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        total_loss_1 += loss_1.item()
        total_loss_2 += loss_2.item()
        total_loss_3 += loss_3.item()
        contr_loss.append(loss_0.detach().cpu().numpy())
        # unmask
        unnorm_out, unnorm_label = unnorm(o1, means, stds), unnorm(l1,means,stds)
        MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)
        
        total_mse.append(MSE.cpu().detach().numpy())
        total_rmse.append(RMSE.cpu().detach().numpy())
        total_mae.append(MAE.cpu().detach().numpy())
        total_mape.append(MAPE.cpu().detach().numpy())
        total_loss.append(loss.item())
    save_print(f'Contrastive loss : {np.mean(contr_loss)}')
    return total_mse,total_rmse, total_mae, total_mape, total_loss, total_loss_1 / (end - start), total_loss_2 / (end - start), total_loss_3 / (end - start)


def test_batch(start,end,model,source_dataset,loss_fn,opt):
    total_loss = []
    total_mae = []
    total_mse = []
    total_rmse = []
    total_mape = []
    total_loss_1, total_loss_2, total_loss_3 = 0, 0, 0
    with torch.no_grad():
        model.eval()
        for idx in trange(start, end, ncols=80):
            data_i, A_wave = source_dataset[idx]
            # [B, N, L, 2]
            x, a, phi, means, stds = data_i.x, data_i.f, data_i.a, data_i.means, data_i.stds
        
            A = torch.squeeze(A_wave).to(args.device).to(torch.float32)
            # remember that the input of TSFormer is [B, N, 2, L]
            
            x = x.permute(0,1,3,2).to(args.device)
            a = a.permute(0,1,3,2).to(args.device)
            phi = phi.permute(0,1,3,2).to(args.device)

            o1, l1, o2, l2, o3, l3, plot_args, en = model([x, a, phi], A)
            
            loss_1 = loss_fn(o1, l1)
            loss_2 = loss_fn(o2, l2)
            loss_3 = loss_fn(o3, l3)
            
            total_loss_1 += loss_1.item()
            total_loss_2 += loss_2.item()
            total_loss_3 += loss_3.item()
            

            unnorm_out, unnorm_label = unnorm(o1, means, stds), unnorm(l1,means,stds)
            MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)

            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())
        return total_mse,total_rmse, total_mae, total_mape, total_loss, total_loss_1 / (end - start), total_loss_2 / (end - start), total_loss_3 / (end - start)

model_path = Path(time.strftime(f'./save/FEPCross_encoder/%m%d-%H%M-') + args.test_dataset + '/')
def save_print(content, folder=str(model_path)):
    with open(f'{folder}/log.out', 'a') as f:
        f.write(str(content))
        f.write('\n')
    print(content)

if __name__ == "__main__":
    if torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
    print("INFO: {}".format(args.device))

    with open(args.config_filename) as f:
        config = yaml.load(f)
        
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    
    if(not os.path.exists(model_path)):
        os.makedirs(model_path)
    data_list = get_data_list(args.data_list)
    
    if args.test_dataset == 'chengdu_m':
        pretrain = 'shenzhen'
    elif args.test_dataset == 'shenzhen':
        pretrain = 'chengdu_m'
    elif args.test_dataset == 'metr-la':
        pretrain = 'pems-bay'
    else:
        pretrain = 'metr-la'
    
    source_dataset = traffic_dataset(data_args, task_args['mae'], data_list, "test", test_data=pretrain, frequency=True)

    model = FEPCross(model_args['mae'], device=args.device).to(args.device)
    model.mode = 'Pretrain'
    
    param_cnt = 0
    for name, param in model.named_parameters():
        print("name: {}, para shape: {}".format(name, param.shape))
        save_print("name: {}, para shape: {}".format(name, param.shape))
        param_cnt += param.numel()
    print("Total parameters : {}".format(param_cnt))
    save_print("Total parameters : {}".format(param_cnt))
    opt = optim.Adam(model.parameters(), lr = task_args['mae']['lr'])

    loss_fn = nn.MSELoss(reduction = 'mean')
    batch_size = task_args['mae']['batch_size']
    save_print('pretrain model has {} parameters'.format(count_parameters(model)))
    
    best_loss = 9999999999999.0
    best_model = None
    for i in range(task_args['mae']['train_epochs']):
        length = source_dataset.__len__()

        train_length = int(train_ratio * length)
        val_length = int(val_ratio * length)
        
        save_print('----------------------')
        time_1 = time.time()
        total_mse,total_rmse, total_mae, total_mape, total_loss, l1, l2, l3 = train_batch(0,train_length, model, source_dataset,loss_fn,opt)
        save_print('Epochs {}/{}'.format(i,task_args['mae']['train_epochs']))
        save_print('Time recons : {}, amplitide recons : {}, phase recons : {}'.format(l1, l2, l3))
        save_print('in training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, normed MSE : {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape),np.mean(total_loss)))
        
        total_mse,total_rmse, total_mae, total_mape, total_loss, l1, l2, l3 = test_batch(train_length,train_length + val_length, model, source_dataset,loss_fn,opt)
        save_print('in validation Time recons : {:.5f}, amplitide recons : {:.5f}, phase recons : {:.5f}'.format(l1, l2, l3))
        save_print('in validation Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))
        
        mae_loss = np.mean(total_mae)
        if(mae_loss < best_loss):
            best_model = model
            best_loss = mae_loss
            torch.save(model.state_dict(), model_path / 'best_model.pt')
            save_print('Best model. Saved.')
        save_print('this epoch costs {:.5}s'.format(time.time()-time_1))
        
        
    total_mse,total_rmse, total_mae, total_mape, total_loss, l1, l2, l3 = test_batch(train_length + val_length,length,model, source_dataset,loss_fn,opt)
    save_print('in test Time recons : {:.5f}, amplitide recons : {:.5f}, phase recons : {:.5f}'.format(l1, l2, l3))
    save_print('test  Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))
    torch.save(model.state_dict(), model_path / 'final_model.pt')