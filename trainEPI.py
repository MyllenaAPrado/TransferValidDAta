import os
import torch
import numpy as np
import logging
import time
import torch.nn as nn
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from model.SwinEPIChannel import IntegratedModelV2
from data.valid2 import VALID_datset
from utils.folders import *
from config import config
from scipy.stats import spearmanr, pearsonr
from torch.utils.tensorboard import SummaryWriter 
from tqdm import tqdm
import torch.optim as optim
from matplotlib import pyplot as plt
from thop import profile

import csv


# Define the RMSE Loss class
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, predicted, actual):
        return torch.sqrt(self.mse(predicted, actual))

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_logging(config):
    if not os.path.exists(config.log_path):
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )

def train_epoch(epoch, net, criterion, optimizer, train_loader,device):
    losses = []
    net.train()
    # save data for one epoch
    pred_epoch = []
    labels_epoch = []
    
    for data in tqdm(train_loader):
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
    logging.info('train epoch:{} / loss:{:.4} / SRCC:{:.4} / PLCC:{:.4}'.format(epoch + 1, ret_loss, rho_s, rho_p))

    return ret_loss, rho_s, rho_p

def eval_model(config, epoch, net, criterion, test_loader,device, i):
    with torch.no_grad():
        losses = []
        net.eval()
        # save data for one epoch
        pred_epoch = []
        labels_epoch = []
        names = []

        for data in tqdm(test_loader):
            
            x_d = data['d_img_org'].to(device)  
            labels = data['score'].to(device)
            name = data['name']

            labels = torch.squeeze(labels.type(torch.FloatTensor)).to(device)  
            start_time = time.time()
            pred = net(x_d)
            end_time = time.time()

            # compute loss
            loss = criterion(torch.squeeze(pred), labels)

            
            logging.info('Image:{} ===== Label:{:.4} ===== Pred:{:.4} '.format(name, labels.item(), pred.item()))

            losses.append(loss.item())

            # save results in one epoch
            pred_batch_numpy = pred.data.cpu().numpy()
            labels_batch_numpy = labels.data.cpu().numpy()
            pred_epoch = np.append(pred_epoch, pred_batch_numpy)
            labels_epoch = np.append(labels_epoch, labels_batch_numpy)
            names = np.append(names, name)
        
        # compute correlation coefficient
        rho_s = abs(spearmanr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0])
        rho_p = abs(pearsonr(np.squeeze(pred_epoch), np.squeeze(labels_epoch))[0])

        path = config.svPath + '/test/{}'.format(config.model_name)
        if not os.path.exists(path):
            os.mkdir(path)
        dataPath = path + '/test_pred_{}.txt'.format(i)
        with open(dataPath, 'w') as f:
            f.write("names, pred_epoch, labels_epoch\n")
            for name, pred, label in zip(names, pred_epoch, labels_epoch):
                f.write(f"{name}, {pred}, {label}\n")
            # Write the statistics
            f.write(f'test epoch: {epoch + 1}  =====  loss: {np.mean(losses):.4f}  =====  SRCC: {rho_s:.4f}  =====  PLCC: {rho_p:.4f} ===== RMSE: {np.mean(losses):.4f}\n')
            
            # Write the time of the epoch
            image_time = end_time - start_time
            f.write(f'Time of an epoch: {image_time}\n')

        logging.info('Epoch:{} ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(epoch + 1, np.mean(losses), rho_s, rho_p))
        return np.mean(losses), rho_s, rho_p, pred_epoch, labels_epoch
    

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    config.log_file = config.model_name + ".log"
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.type_name)
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.model_name)

    config.ckpt_path = os.path.join(config.ckpt_path, config.type_name)
    config.ckpt_path = os.path.join(config.ckpt_path, config.model_name)

    config.log_path = os.path.join(config.log_path, config.type_name)

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    
    if not os.path.exists(config.tensorboard_path):
        os.makedirs(config.tensorboard_path)

    set_logging(config)
    logging.info(config)

    writer = SummaryWriter(config.tensorboard_path)

    dataset = config.dataset
    i=0


    for train_folders, val_folder, test_folders in k_folders():
        if dataset == "VALID":
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
            train_dataset = VALID_datset(folders=train_folders, transform=transform_train)        
            val_dataset = VALID_datset(folders=val_folder, transform=transform_eval)
            test_dataset = VALID_datset(folders=test_folders, transform=transform_eval)

        elif (dataset == 'WIN'):
            
            transform_train=transforms.Compose([
                            transforms.CenterCrop((3360, 512)),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomRotation(15),
                            transforms.ToTensor()
                        ])
            transform_eval =transforms.Compose([
                            transforms.CenterCrop((3360, 512)),
                            transforms.ToTensor()
                        ]) 

            # data load
            train_dataset = Win5LID_datset(folders=train_folders, transform=transform_train)        
            val_dataset = Win5LID_datset(folders=val_folder, transform=transform_eval)
            test_dataset = Win5LID_datset(folders=test_folders, transform=transform_eval)
       
        elif (dataset == 'LFDD'):
            
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
            train_dataset = LFDD_datset(folders=train_folders, transform=transform_train)        
            val_dataset = LFDD_datset(folders=val_folder, transform=transform_eval)
            test_dataset = LFDD_datset(folders=test_folders, transform=transform_eval)
        
        # load the data
        train_loader = DataLoader(dataset=train_dataset+val_dataset, batch_size=config.batch_size, shuffle=True)
        
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        #create model
        # Parameters
        image_size = (3360, 512)  # Input image size
        in_channels = 3  # RGB image
        patch_size = 4
        emb_size = 96
        reduction_ratio = 20
        swin_window_size = [8,8,4]
        num_heads = [2,3,2]
        swin_blocks = [2,2,2]

        # Initialize the model
        model = IntegratedModelV2(image_size=image_size, in_channels=in_channels, 
                              patch_size=patch_size, emb_size=emb_size, 
                              reduction_ratio=reduction_ratio, swin_window_size=swin_window_size, 
                              num_heads=num_heads, swin_blocks=swin_blocks,
                              num_stb=3)
    
        model = model.to(device)

        ### Create three input tensors, each with shape (1, 3, 224, 224)
        input_tensor = torch.randn(1, 3, 3360, 512).to(device)  # Example input tensor
        flops, params = profile(model, inputs=(input_tensor,))
        logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), model.parameters())) / 10 ** 6))
        logging.info('Flops: {} '.format(flops))

        criterion = RMSELoss() 
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0001,
        )
        # Learning rate scheduler to halve the learning rate every 200 epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


        # train & validation
        losses, scores = [], []
        best_srocc = 0
        best_plcc = 0
        main_score = 0
        loss_global = 999
        count_improvment = 0
        loss_plot = []

        for epoch in range(0, config.n_epoch):
            start_time = time.time()
            logging.info('Running training epoch {}'.format(epoch + 1))
            loss_val, rho_s, rho_p = train_epoch(epoch, model, criterion, optimizer, train_loader, device)

            writer.add_scalar("Train_loss", loss_val, epoch)
            writer.add_scalar("SRCC", rho_s, epoch)
            writer.add_scalar("PLCC", rho_p, epoch)
            logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))
            scheduler.step()

        plt.clf() 
        plt.figure()
        plt.plot(loss_plot)
        plt.title("Training and Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(["Training", "Validation"])
        plt.savefig(f'TrainingLossCurve{dataset}.png')
        
        loss, rho_s, rho_p, pred, labels = eval_model(config=config, epoch=epoch, net= model, criterion=criterion, 
                                                      test_loader=test_loader, device=device, i=i)
        logging.info('Result ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(loss, rho_s, rho_p))

        print('Result ===== loss:{:.4} ===== SRCC:{:.4} ===== PLCC:{:.4}'.format(loss, rho_s, rho_p))




        plt.clf() 
        plt.scatter(np.array(labels), np.array(pred))

        # Plot the diagonal line
        diagonal_line = np.linspace(min(min(labels-1), min(pred-1)), max(max(labels+1), max(pred+1)), 100)
        plt.plot(diagonal_line, diagonal_line, 'r--', label='y = x')

        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')	
        plt.savefig(f'ActualPredicted{dataset}.png')

        i+=1
        