import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms



class VALID_datset(torch.utils.data.Dataset):
    def __init__(self,  folders, transform):
        super(VALID_datset, self).__init__()
        
        self.dataset_dic = self.get_paths_score(folders)
        self.transform = transform
        
    def get_paths_score(self, folders):
        root='data/centralValid/' #datasetVALID'
        df = pd.read_csv("data/passive_scores.csv")
        data_dic = {}
        idx=0
        for folder in folders:
            if folder == 'I':
                root_images= os.path.join(root,folders)
                for file in os.listdir(root_images):
                    file_name = file.split('.')[0]  
                    if '_' in file_name:
                        value = df[file_name.split('_')[1]].sum()/len(df[file_name.split('_')[1]])
                    else:
                        value = df[file_name.split('_')[0]].sum()/len(df[file_name.split('_')[0]])
                    file_path = os.path.join(root_images,file)
                    data_dic[idx]= (file_path, value, file_name)

                    idx+=1
                break
            else:            
                root_images= os.path.join(root,folder)
                for file in os.listdir(root_images):
                    file_name = file.split('.')[0]  
                    if '_' in file_name:
                        value = df[file_name.split('_')[1]].sum()/len(df[file_name.split('_')[1]])
                    else:
                        value = df[file_name.split('_')[0]].sum()/len(df[file_name.split('_')[0]])
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

        images = []
        for img in os.listdir(d_img_path):
            if(img == 'mli.png'):
                mli_img = Image.open(f'{d_img_path}/{img}').convert('RGB')
                mli_img= transforms.ToTensor()(mli_img)
                #if self.transform:
                #    mli_img = self.transform(mli_img)

            else:
                if '009' not in img:
                    image = Image.open(f'{d_img_path}/{img}').convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    images.append(image)        

        images = torch.stack(images) 
        sample = {
            'd_img_org': images,
            'score': score,
            'mli': mli_img,
            'name': name
        }        

        return sample


    
    