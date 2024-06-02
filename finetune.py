import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from dataset import traffic_dataset
from utils import *
import argparse
import yaml
import time
import sys
sys.path.append('./model')
sys.path.append('./model/FEPCross_models')
sys.path.append('./model/STmodels')
sys.path.append('./model/combined_models')
from gwn import *
from FEPCross import *
from contrastive_loss import *
from combined_model import *
from final_model import *
import random


parser = argparse.ArgumentParser(description='TPB')
parser.add_argument('--config_filename', default='./configs/config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='chengdu_m', type=str)
parser.add_argument('--train_epochs', default=200, type=int)
parser.add_argument('--finetune_epochs', default=120,type=int)
parser.add_argument('--lr',default=1e-3,type=float)
parser.add_argument('--decay',default=0.9,type=float)
parser.add_argument('--update_step', default=5,type=int)
parser.add_argument('--momentum_ratio', default=0.9,type=float)
parser.add_argument('--seed',default=7,type=int)
parser.add_argument('--data_list', default='chengdu_shenzhen_metr',type=str)
parser.add_argument('--target_days', default=3,type=int)
parser.add_argument('--patch_encoder', default='pattern', type=str)
parser.add_argument('--gpu', default=0, type = int)
parser.add_argument('--sim', default='cosine', type = str)
parser.add_argument('--K', default=10, type = int)
parser.add_argument('--epochs', default=100, type = int)
parser.add_argument('--meta_epochs', default=100, type = int)
parser.add_argument('--STmodel',default='GWN',type=str)
parser.add_argument('--en_trainable',default=0,type=int)
parser.add_argument('--baseline',default=0,type=int)
parser.add_argument('--moredata',default=1,type=int)
parser.add_argument('--fake_ratio',default=0.1,type=float)
args = parser.parse_args()

args.new=1
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float32)

# since historical 1 day data is used to generate metaknowledge
folder = time.strftime(f'./save/FEPCross_model/%m%d-%H%M-') + args.test_dataset
def save_print(content, folder=folder):
    with open(f'{folder}/log.out', 'a') as f:
        f.write(str(content))
        f.write('\n')
    print(content)

if __name__ == '__main__':
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    #save_print("Forecasting target_days = {}".format(args.target_days - 1))
    
    if torch.cuda.is_available():
        args.device = torch.device(f'cuda:{args.gpu}')
        save_print("INFO: GPU : {}".format(args.gpu))
    else:
        args.device = torch.device('cpu')
        save_print("INFO: CPU")

    with open(args.config_filename) as f:
        config = yaml.load(f)

    config['model']['mae']['mask_ratio'] = 0.05

    args.data_list = config['model']['STnet']['data_list']
    args.batch_size = config['task']['maml']['batch_size']
    #args.test_dataset = args.en1
    args.K = config['model']['STnet']['K']
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    data_list = get_data_list(args.data_list)
    save_print("INFO: finetuning on {}.".format(args.test_dataset))
    save_print(args)
    
    save_print(data_args)
    save_print(task_args)
    save_print(model_args)
    

    encoder = FEPCross(model_args['mae'], device=args.device).to(args.device)
    encoder.mode = 'Finetune'
    
    encoder.load_state_dict(torch.load(f'./save/{args.test_dataset}/best_model.pt', map_location=f'cuda:{args.gpu}'))
    en_trainable = False

    
    if args.baseline == 0:
        baseline = False
    else:
        baseline = True
    
    if args.test_dataset == 'chengdu_m':
        N = 524
    elif args.test_dataset == 'shenzhen':
        N = 627
    elif args.test_dataset == 'metr-la':
        N = 207
    else:
        N = 325
    
    model = FEPCross_wrap(encoders=[encoder], gwn=None, device=args.device, folder=folder, epochs=60, en_trainable=en_trainable,model_args = model_args,args = args, baseline=baseline, node_num=N)
    
    
    test_time_dataset = traffic_dataset(data_args, task_args['maml'], data_list, "test", test_data=args.test_dataset, target_days=2, frequency=True)
    
    time_dataset = traffic_dataset(data_args, task_args['maml'], data_list, "target", test_data=args.test_dataset, target_days=2, frequency=True)
    time_dataset.generate_fake(encoder, args.device, args.fake_ratio, args.moredata)
    
    model.finetune(time_dataset, args.epochs)
    model.test(test_time_dataset)

    torch.save(model.state_dict(), f'{folder}/final_encoder.pt')
    
    
