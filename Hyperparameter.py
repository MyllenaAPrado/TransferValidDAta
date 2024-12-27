#/bin/bash
# -*- coding: utf-8 -*-
##########################################################################################################
import torch
import optuna
import torch
from optuna.trial import TrialState
import numpy as np 
from torch_geometric.loader import DataLoader
from scipy.stats import spearmanr,  pearsonr
import os
import pandas as pd
import torch.nn as nn
from data.win5lid import Win5LID_datset
from data.valid2 import VALID_datset
from torchvision import transforms

from model.SwinEPIChannel import IntegratedModelV2
from torch.utils.data import DataLoader

# Define the RMSE Loss class
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predicted, actual):
        return torch.sqrt(self.mse(predicted, actual))
    

def train_epoch(epoch, net, criterion, optimizer, train_loader,device):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in train_loader:
        x_d = data['d_img_org'].to(device)
        labels = data['score'].to(device)

        labels = torch.squeeze(labels.type(torch.FloatTensor)).to(device)  
        pred_d = net(x_d)

        optimizer.zero_grad()
        loss = criterion(torch.squeeze(pred_d), labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()
        # save results in one epoch
        pred_batch_numpy = pred_d.data.cpu().numpy()
        labels_batch_numpy = labels.data.cpu().numpy()
        pred_epoch = np.append(pred_epoch, pred_batch_numpy)
        labels_epoch = np.append(labels_epoch, labels_batch_numpy)
    
    # compute correlation coefficient
    rho_s = abs(spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0])
    rho_p = abs(pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0])

    ret_loss = np.mean(losses)
    #logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss

def eval_epoch(epoch, net, criterion, test_loader,device):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []

        for data in test_loader:
            
            x_d = data['d_img_org'].to(device)  
            labels = data['score'].to(device)
            labels = torch.squeeze(labels.type(torch.FloatTensor)).to(device)  
            pred = net(x_d)

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)
            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
        
        # compute correlation coefficient
        rho_s = abs(spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0])
        rho_p = abs(pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0])

        #logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        return np.mean(losses), rho_p, rho_s


def objective(trial):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform_train=transforms.Compose([
                    transforms.CenterCrop((3360, 512)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor()
                ])
    transform_eval =transforms.Compose([
                    transforms.CenterCrop((3360, 512)),
                    transforms.ToTensor()
                ]) 

    # data load
    train_dataset = Win5LID_datset(folders=["Bikes", "Flowers","museum", "rosemary", "Swans", "Vespa", "dishes", "greek"], transform=transform_train)        
    test_dataset = Win5LID_datset(folders=[ "Palais", "Sphynx"], transform=transform_eval, print_folder=True)
    #train_dataset = VALID_datset(folders=["I02", "I04", "I09"], transform=transform_train)        
    #test_dataset = VALID_datset(folders=["I01", "I10"], transform=transform_eval)
    
    # Hyperparameters to be optimized
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float('wd', 1e-4, 1e-1, log=True) 
    batch_size = trial.suggest_categorical('batch_size', [2, 4, 6, 8])

    #fcc_2 = trial.suggest_categorical('fcc_2', [32, 64, 128, 256, 512])
    
    image_size = (3360, 512)
    in_channels = 3
    patch_size =  trial.suggest_categorical('patch_size', [2, 4, 6, 8])
    emb_size = trial.suggest_categorical('emb_size', [32, 64, 96, 128, 256])
    reduction_ratio = trial.suggest_categorical('reduction_ratio', [8, 12, 16, 20, 24, 32])

    swin_window_size = [
        trial.suggest_categorical('swin_window_size_0', [2, 4, 6, 8]),  # Vary the first element between 2 and 8
        trial.suggest_categorical('swin_window_size_1', [2, 4, 6, 8]),  # Vary the second element between 2 and 8
        trial.suggest_categorical('swin_window_size_2', [2, 4, 6, 8])   # Vary the third element between 2 and 8
    ]

    num_heads = [
        trial.suggest_categorical('num_heads_0', [2, 4, 6]),  # Vary the first element between 2 and 8
        trial.suggest_categorical('num_heads_1', [2, 4, 6]),  # Vary the second element between 2 and 8
        trial.suggest_categorical('num_heads_2', [2, 4, 6])   # Vary the third element between 2 and 8
    ]

    swin_blocks = [
        trial.suggest_int('swin_blocks_0', 2, 4),  # Vary the first element between 2 and 8
        trial.suggest_int('swin_blocks_1', 2, 4),  # Vary the second element between 2 and 8
        trial.suggest_int('swin_blocks_2', 2, 6)   # Vary the third element between 2 and 8
    ]

    EAM = trial.suggest_categorical('eam', [True, False])
    num_stb = trial.suggest_categorical('num_stb', [1, 2, 3])

    model = IntegratedModelV2(image_size=image_size, in_channels=in_channels, 
                              patch_size=patch_size, emb_size=emb_size, 
                              reduction_ratio=reduction_ratio, swin_window_size=swin_window_size, 
                              num_heads=num_heads, swin_blocks=swin_blocks,
                              EAM=EAM, num_stb=num_stb)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)

    loss_func_options = {
        'mse': nn.MSELoss(),
        'rmse': RMSELoss() 
    }

    # Optuna chooses from the list of loss functions
    loss_func_name = trial.suggest_categorical('loss_func', list(loss_func_options.keys()))
    loss_function = loss_func_options[loss_func_name]

    # load the data
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


    try: 
        # Training
        for epoch in range(30):  # Number of epochs can also be a hyperparameter
            avg_loss = train_epoch(epoch, model, loss_function, optimizer, train_loader, device)
    
        # Testing
        test_loss, plcc, srcc = eval_epoch(epoch, model, loss_function, test_loader, device)
        print(plcc, srcc)
   
    except:
        test_loss = 999999999999
        
    return -test_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100) #, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
