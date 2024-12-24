import os
import json
import random
import torch
import json
import numpy as np
from loguru import logger
from sklearn.metrics import f1_score, roc_auc_score


def set_random_seed(seed, device):
    # for reproducibility (always not guaranteed in pytorch)
    # [1] https://pytorch.org/docs/stable/notes/randomness.html
    # [2] https://hoya012.github.io/blog/reproducible_pytorch/

    if device == 'cpu':
        pass
    else:
        device = device.split(':')[0]

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            logger.info('The seed is set up to device!')


def log_param(param):
    for key, value in param.items():
        if type(value) is dict:
            for in_key, in_value in value.items():
                logger.info('{:20}:{:>50}'.format(
                    in_key, '{}'.format(in_value)))
        else:
            logger.info('{:20}:{:>50}'.format(key, '{}'.format(value)))


def export_setting(path, in_dict: dict):
    in_dict['device'] = str(in_dict['device'])

    with open(f'{path}/settings.json', 'w') as f:
        json.dump(in_dict, f)


def load_model_config(file_path):
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as json_config:
            param = json.load(json_config)
        return param
    else:
        raise Exception("No config file")


def save_model_config(param):
    json_path = f"../config/{param['model']}/{param['epochs']}-{param['seed']}-{param['num_layer']}-{param['dataset_name']}-{param['seed']}.json"

    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path))

    with open(json_path, 'w', encoding='utf-8') as json_config:
        json.dump(param, json_config, ensure_ascii=False, indent=4)


def convert_array_to_tensor_in_dict(dict, device):
    for key, value in dict.items():
        if type(value) is np.ndarray:
            dict[key] = torch.tensor(value, device=device)
        elif type(value) is type({}):
            for k, v in value.items():
                if type(v) == np.ndarray:
                    dict[key][k] = torch.tensor(v, device=device)
                elif type(v) == torch.Tensor:
                    dict[key][k] = v.to(device)
        elif type(value) is torch.Tensor:
            dict[key] = dict[key].to(device)
        elif type(value) is list:
            dict[key] = [i.to(device) for i in value]
    return dict


def get_normalized_matrices(edges, sign, num_nodes):
    """
    Normalized signed adjacency matrix

    :param edges: signed edges
    :param num_nodes: number of nodes
    :return: normalized matrices
    """
    sign = sign.clone()
    sign[sign == 0] = -1
    
    row, col, data = edges[:, 0], edges[:, 1], sign
    shaping = (num_nodes, num_nodes)
    
    A = torch.sparse_coo_tensor(torch.stack([row, col]), data, shaping)
    A = torch.eye(num_nodes, device=A.device).to_sparse() + A
    
    rowsum = torch.abs(A).sum(dim=1).to_dense()
    rowsum[rowsum == 0] = 1  # avoid division by zero
    
    r_inv = torch.pow(rowsum, -1).flatten()
    r_mat_inv = torch.diag(r_inv).to_sparse()
    
    snA = r_mat_inv @ A
    pos_idx, neg_idx = snA.values() > 0, snA.values() < 0
    
    nApT_idx = snA.indices()[:, pos_idx]
    nAmT_idx = snA.indices()[:, neg_idx]
    
    nApT = torch.sparse_coo_tensor(
        nApT_idx, snA.values()[pos_idx], snA.size()).t()
    nAmT = torch.sparse_coo_tensor(nAmT_idx, torch.abs(
        snA.values()[neg_idx]), snA.size()).t()

    return nApT, nAmT


def metric(truths, probs):
    """calculate metric

    Args:
        truths (torch.tensor): label
        preds (torch.tensor): preds

    Returns:
        pred_dict: {auroc, f1-bi, f1-mi, f1-ma}
    """
    truths = truths.squeeze().cpu().detach().numpy()
    
    preds = (probs >= 0.5).long().squeeze().cpu().detach().numpy()
    probs = probs.squeeze().cpu().detach().numpy()

    return {"auc": roc_auc_score(truths, probs), "f1-bi": f1_score(truths, preds, average='binary'), "f1-mi": f1_score(truths, preds, average='micro'), "f1-ma": f1_score(truths, preds, average='macro')}


