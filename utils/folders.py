from config import config
from itertools import combinations
import numpy as np

def generate_datasets(name_folders, k=10):
    """
    Generates datasets for k-fold cross-validation with combinations for testing.
    Each dataset consists of train, val, and test splits.

    Parameters:
        name_folders (list): List of folder names.
        k (int): Number of folders (length of name_folders).

    Returns:
        list of tuples: Each tuple contains (train_folders, val_folder, test_folders).
    """
    name_folders = list(name_folders)
    datasets = []

    # Generate all combinations of two folders for testing
    for i, j in combinations(range(k), 2):
        test_folders = [name_folders[i], name_folders[j]]
        
        # Remaining folders after selecting test folders
        remaining_folders = [name_folders[idx] for idx in range(k) if idx not in (i, j)]
        
        # Randomly select a validation folder from the remaining folders
        val_folder = remaining_folders[np.random.randint(len(remaining_folders))]
        
        # Remaining folders for training
        train_folders = [folder for folder in remaining_folders if folder != val_folder]
        
        # Append the dataset configuration
        datasets.append((train_folders, val_folder, test_folders))
    
    return datasets


def k_folders():


    if config.dataset == 'WIN':
        name_folders = {"Bikes", "Flowers", "Palais", "Sphynx", "Swans", "Vespa", "dishes", "greek", "museum", "rosemary"}

        folders = generate_datasets(name_folders, k=10)

    if config.dataset == 'VALID':
        #name_folders = {'I01', 'I02', 'I04', 'I09', 'I10'}
        name_folders = {
            'I01', 'I01_horizontal_flip', 'I01_vertical_flip', #'I01_vertical_horizontal_flip', 
            'I02', 'I02_horizontal_flip', 'I02_vertical_flip', #'I02_vertical_horizontal_flip', 
            'I04', 'I04_horizontal_flip', 'I04_vertical_flip', #'I04_vertical_horizontal_flip', 
            'I09', 'I09_horizontal_flip', 'I09_vertical_flip', #'I09_vertical_horizontal_flip', 
            'I10', 'I10_horizontal_flip', 'I10_vertical_flip', #'I10_vertical_horizontal_flip', 
            #'vI01', 'vI01_horizontal_flip', 'vI01_vertical_flip', 'vI01_vertical_horizontal_flip', 
            #'vI02', 'vI02_horizontal_flip', 'vI02_vertical_flip', 'vI02_vertical_horizontal_flip', 
            #'vI04', 'vI04_horizontal_flip', 'vI04_vertical_flip', 'vI04_vertical_horizontal_flip', 
            #'vI09', 'vI09_horizontal_flip', 'vI09_vertical_flip', 'vI09_vertical_horizontal_flip', 
            #'vI10', 'vI10_horizontal_flip', 'vI10_vertical_flip', 'vI10_vertical_horizontal_flip'
        }

        folders = generate_datasets(name_folders, k=15)

    if config.dataset == 'VALID8':
        name_folders = {'I01', 'I02', 'I04', 'I09', 'I10'}

        folders = generate_datasets(name_folders, k=5)

    if config.dataset == 'LFDD':
        name_folders = {'boxes', 'cotton', 'kitchen', 'pens', 'pillows', 'rosemary', 'sideboard', 'vinyl'}

        folders = generate_datasets(name_folders, k=8)


    return folders



