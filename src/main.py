import os
import csv
import sys
import torch
from loguru import logger
from fire import Fire
from data_loader.DataLoader import DataLoader
from utils import log_param, set_random_seed, load_model_config, save_model_config
from Trainer import Trainer

def main(param):
    """
    This method operates the overall procedures.

    Args:
        param (dict): the dictionary of mapped options

    Raises:
        Exception: if do not support dataset
    """
    # Step 0. Initialization 
    device = param["device"] if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=param["seed"], device=device)
    log_param(param)
    save_model_config(param)

    # Step 1. Preprocessing the dataset
    dataset_list = ["review", "ml-1m", "bonanza", "amazon-dm"]
    if param['dataset_name'].lower() not in dataset_list:
        raise Exception("not supported dataset")
    data_loader = DataLoader(**param)
    pre_processed_data, param["num_nodes"] = data_loader.get_data(device)
        
    # Step 2. Model Train
    result_list = Trainer(pre_processed_data, **param).train_model()
    
    # Step 3. Report the result
    best_valid_epoch_auc = -1
    best_valid_epoch_f1 = -1
    best_valid_score_auc = -float('inf')
    best_valid_score_f1 = -float('inf')
    
    for epoch, i in enumerate(result_list):
        if i["valid"]["auc"] > best_valid_score_auc:
            best_valid_score_auc = i["valid"]["auc"]
            best_valid_epoch_auc = epoch
        if i["valid"]["f1-ma"] > best_valid_score_f1:
            best_valid_score_f1 = i["valid"]["f1-ma"]
            best_valid_epoch_f1 = epoch
    
    print(f"Best valid score auc: {round(best_valid_score_auc,3)}, Best test score auc: {round(result_list[best_valid_epoch_auc]['test']['auc'],3)}")
    print(f"Best valid score macro f1: {round(best_valid_score_f1,3)}, Best test score macro f1: {round(result_list[best_valid_epoch_f1]['test']['f1-ma'],3)}")
    
def main_wraper(dataset="review", config=None, **kwargs):
    """
    This method configures simulation from given option.
    
    Args:
        dataset (str, optional): dataset name.
        config (str, optional): 
            if config is directory path 
                load the config file. 
            else 
                use the default config file. 
                overwrite the config file with kwargs if config is None.

    Raises:
        ValueError: if not supported option
    """
    
    if config is not None:
        if kwargs:
            raise ValueError('config and kwargs cannot be used together!')
        config = load_model_config(config)
    
    else:
        base_config_path = f"../config/base_config/{dataset.lower()}.json"
        config = load_model_config(base_config_path)
        for k in kwargs:
            if k in config:
                logger.warning('{} will be overwritten!'.format(k))
                config[k] = kwargs[k]
            else:
                raise ValueError
    main(config)

if __name__ == "__main__":
    Fire(main_wraper)
