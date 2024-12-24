# template for data loader
import numpy as np
from data_loader.utils import split_data, load_data, split_edges_by_sign, make_degree_inv, normalization, svd
from utils import convert_array_to_tensor_in_dict
import torch

class DataLoader(object):
    """Template for data loader

    Args:
        model (str): Model
        dataset_name (str): dataset name
        seed (int): seed
        split_ratio (list): [train(float), val(float), test(float)], train+val+test == 1 
        dataset_shuffle (bool): dataset_shuffle if True
        device (str): device
        direction (str): True-direct, False-undirect
        node_idx_type (str): "uni" or "bi"
    """

    def __init__(
        self,
        model: str,
        dataset_name: str,
        seed: int,
        split_ratio: list,
        dataset_shuffle: bool,
        device: str,
        direction: bool,
        node_idx_type: str,
        input_dim: int,
        **kwargs
    ) -> None:
        self.model = model
        self.dataset_name = dataset_name
        self.dataset_path = f"../datasets/{self.dataset_name}.tsv"
        self.seed = seed
        self.split_ratio = split_ratio
        self.dataset_shuffle = dataset_shuffle
        self.device = device
        self.direction = direction
        self.node_idx_type = node_idx_type
        self.input_dim = input_dim
        assert node_idx_type.lower() in [
            "uni", "bi"], "not supported node_idx_type"
        assert np.isclose(sum(split_ratio), 1).item(
        ), "sum of split_ratio is not 1"
        self.processing(**kwargs)

    def processing(
        self,
        **kwargs
    ):
        array_of_edges, self.num_nodes, self.num_edges = load_data(
            self.dataset_path, self.direction, self.node_idx_type)
        
        processed_dataset = split_data(
            array_of_edges, self.split_ratio, self.seed, self.dataset_shuffle)
        processed_dataset["num_nodes"] = self.num_nodes
        
        augmented_dataset = self.augment_graph(processed_dataset["train_edges"], processed_dataset["train_label"], model=self.model, **kwargs)
        if augmented_dataset != None:  
            for key in augmented_dataset:
                if key in processed_dataset:
                    print(f"{key} is overwirted when dataloading")
                processed_dataset[key] = augmented_dataset[key]
        processed_dataset["init_emb"] = self.set_init_embeddings() 
        
        self.processed_dataset = processed_dataset

    def get_data(
        self,
        device
    ):  
        
        return convert_array_to_tensor_in_dict(self.processed_dataset, device=device), self.num_nodes

    def set_init_embeddings(self):
        """
        set embeddings function for training model

        Args:
            embeddings (torch.Tensor): embeddings
        """
        if self.node_idx_type == "uni":
            embeddings = torch.nn.init.xavier_uniform_(torch.empty(
                (sum(self.num_nodes), self.input_dim)))
            return embeddings
        elif self.node_idx_type == "bi":
            self.embeddings_user = torch.nn.init.xavier_uniform_(
                torch.empty(self.num_nodes[0], self.input_dim))
            self.embeddings_item = torch.nn.init.xavier_uniform_(
                torch.empty(self.num_nodes[1], self.input_dim))
            return [self.embeddings_user, self.embeddings_item]
        
    def augment_graph(
        self,
        train_edge: np.array,
        train_label: np.array,
        rank_ratio: float,
        **kwargs
    ) -> np.array:
        """ licos augmentation graph processor

        Args:
            train_edge (np.array): _description_
            train_label (np.array): _description_

        Returns:
            train_ind_label (dict):
            A_pos (tensor)
            A_neg (tensor)
            B_pos (tensor)
            B_neg (tensor)
            svd_A (dict)
            svd_B (dict)
        """
        assert rank_ratio <= 1 and rank_ratio > 0, "rank_ratio between 0 and 1"
        self.rank = int(min(self.num_nodes) * rank_ratio)
        assert self.rank >= 1, "rank must bigger than 1"
        pos_train_edges, neg_train_edges = split_edges_by_sign(
            train_edge, train_label, False)

        u_train = pos_train_edges[0] + neg_train_edges[0]
        v_train = pos_train_edges[1] + neg_train_edges[1]
        y_label_train = [1 * len(pos_train_edges[0])] + \
            [0 * len(neg_train_edges[0])]
        train_ind_label = {'u_train': u_train,
                           'v_train': v_train, 'y_label_train': y_label_train}

        R_pos = torch.sparse_coo_tensor(pos_train_edges, [
                                        1]*len(pos_train_edges[0]), size=self.num_nodes, dtype=torch.float)
        R_neg = torch.sparse_coo_tensor(
            neg_train_edges, [1]*len(neg_train_edges[0]), size=self.num_nodes, dtype=torch.float)

        row_degree_inv, col_degree_inv = make_degree_inv(R_pos + R_neg)
        A_pos, A_neg, B_pos, B_neg = normalization(
            self.device, R_pos, R_neg, row_degree_inv, col_degree_inv)
        del R_pos, R_neg, row_degree_inv, col_degree_inv

        svd_A = svd(self.device, A_pos, A_neg, self.rank)
        svd_B = svd(self.device, B_pos, B_neg, self.rank)

        return {"train_ind_label": train_ind_label, "A_pos": A_pos, "A_neg": A_neg, "B_pos": B_pos, "B_neg": B_neg, "svd_A": svd_A, "svd_B": svd_B}
