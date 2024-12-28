import os
import torch
import numpy as np
import pandas as pd
from PIL import Image



class LFDD_datset(torch.utils.data.Dataset):
    def __init__(self,  folders, transform, print_folder=False):
        super(LFDD_datset, self).__init__()
        
        self.dataset_dic = self.get_paths_score(folders, print_folder)
        self.transform = transform
        
    def get_paths_score(self, folders, print_folder=False):
        root='data/LFDDHor'
        df = pd.read_csv("data/mos.csv")
        data_dic = {}
        idx=0

        if isinstance(folders, list):
            for folder in folders:
                root_images= os.path.join(root,folder)
                for file in os.listdir(root_images):
                    file_name = file.split('.')[0] 
                    value = df.loc[df['folder'] == file_name]['Sum'].values[0]    
                    file_path = os.path.join(root_images,file)
                    data_dic[idx]= (file_path, value, file_name)
                    idx+=1

        else:
            root_images= os.path.join(root,folders)
            for file in os.listdir(root_images):
                file_name = file.split('.')[0]  
                value = df.loc[df['folder'] == file_name]['Sum'].values[0]    
                file_path = os.path.join(root_images,file)
                data_dic[idx]= (file_path, value, file_name)
                idx+=1

        return data_dic

    def normalization(self, data):
        range = np.max(data) - np.min(data)
        return (data - np.min(data)) / range

    def __len__(self):
        return len(self.dataset_dic)

    def __getitem__(self, idx):

        d_img_path, score, name = self.dataset_dic.get(idx, 0)           
        image = Image.open(f'{d_img_path}')
        if self.transform:
            image = self.transform(image)  
       
        sample = {
            'd_img_org': image,
            'score': score,
            'name': name
        }        

        return sample




    
    