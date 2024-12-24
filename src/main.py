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
    result_list = Trainer(pre_processed_data, **param).train()
    
def main_wraper(model="elise", conf_file=None, **kwargs):
    conf_file = kwargs.get('conf_file', None)
    if conf_file is not None:
        config = load_model_config(conf_file)
        for k in kwargs:
            if k in config:
                logger.warning('{} will be overwritten!'.format(k))
                config[k] = kwargs[k]
            else:
                raise ValueError
        main(config)
    else:
        base_config_path = f"../config/base_config/{model.lower()}.json"
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
