import torch
import numpy as np
from sklearn.model_selection import train_test_split



def split_data(
    array_of_edges: np.array,
    split_ratio: list,
    seed: int,
    dataset_shuffle: bool,
) -> dict:
    """Split your dataset into train, valid, and test

    Args:
        array_of_edges (np.array): array_of_edges
        split_ratio (list): train:test:val = [float, float, float], train+test+val = 1.0 
        seed (int) = seed
        dataset_shuffle (bool) = shuffle dataset when split

    Returns:
        dataset_dict: {train_edges : np.array, train_label : np.array, test_edges: np.array, test_labels: np.array, valid_edges: np.array, valid_labels: np.array}
    """

    assert np.isclose(sum(split_ratio), 1), "train+test+valid != 1"
    train_ratio, valid_ratio, test_ratio = split_ratio
    train_X, test_val_X, train_Y, test_val_Y = train_test_split(
        array_of_edges[:, :2], array_of_edges[:, 2], test_size=1 - train_ratio, random_state=seed, shuffle=dataset_shuffle)
    val_X, test_X, val_Y, test_Y = train_test_split(test_val_X, test_val_Y, test_size=test_ratio/(
        test_ratio + valid_ratio), random_state=seed, shuffle=dataset_shuffle)

    dataset_dict = {
        "train_edges": train_X,
        "train_label": train_Y,
        "valid_edges": val_X,
        "valid_label": val_Y,
        "test_edges": test_X,
        "test_label": test_Y
    }

    return dataset_dict


def load_data(
    dataset_path: str
) -> np.array:
    """Read data from a file

    Args:
        dataset_path (str): dataset_path

    Return:
        array_of_edges (array): np.array of edges
        num_of_nodes: [type1(int), type2(int)]
    """

    edgelist = []
    with open(dataset_path) as f:
        for line in f:
            a, b, s = map(int, line.split('\t'))
            if s == -1:
                s = 0
            edgelist.append((a, b, s))
    num_of_nodes = get_num_nodes(np.array(edgelist))
    edgelist = np.array(edgelist)
    num_edges = np.array(edgelist).shape[0]

    return edgelist, num_of_nodes, num_edges


def get_num_nodes(
    dataset: np.array
) -> int:
    """get num nodes when bipartite

    Args:
        dataset (np.array): dataset

    Returns:
        num_nodes tuple(int, int): num_nodes_user, num_nodes_item
    """
    num_nodes_user = np.amax(dataset[:, 0]) + 1
    num_nodes_item = np.amax(dataset[:, 1]) + 1
    return (num_nodes_user.item(), num_nodes_item.item())


def split_edges_by_sign(
    edge_lists: np.array,
    edge_label: np.array,
    consider_direction: bool
) -> list:
    """split edge list by sign

    Args:
        edge_lists (np.array): edge_array
        edge_label (np.array): edge_sign
        consider_direction (bool): 
            False : return edgelist_pos, edgelist_neg
            True : return edgelist_pos_u_i, edgelist_neg_u_i, edgelist_i_u, edgelist_i_u

    Returns:
        pos_edges (list): pos_edges
        neg_edges (list): neg_edges
    """

    edgelist_pos_u_i, edgelist_neg_u_i = [[], []], [[], []]
    edgelist_pos_i_u, edgelist_neg_i_u = [[], []], [[], []]

    for (fr, to), sign in zip(edge_lists.tolist(), edge_label.tolist()):
        if sign == 1:
            edgelist_pos_u_i[0].append(fr)
            edgelist_pos_u_i[1].append(to)
            edgelist_pos_i_u[0].append(to)
            edgelist_pos_i_u[1].append(fr)

        elif sign == 0:
            edgelist_neg_u_i[0].append(fr)
            edgelist_neg_u_i[1].append(to)
            edgelist_neg_i_u[0].append(to)
            edgelist_neg_i_u[1].append(fr)

        else:
            print(fr, to, sign)
            raise Exception("sign must be 0/1")
    if consider_direction:
        return edgelist_pos_u_i, edgelist_neg_u_i, edgelist_pos_i_u, edgelist_neg_i_u
    else:
        return edgelist_pos_u_i, edgelist_neg_u_i


def make_degree_inv(
    R: torch.sparse_coo_tensor
) -> torch.tensor:
    """ get degree inverse

    Args:
        R (torch.sparse_coo_tensor): bi-adjacency matrix

    Returns:
        u_diag_inv_list (tnesor): inversed degree matrix col
        v_diag_int_list (tensor): inversed degree matrix row
    """

    abs_R = abs(R)
    u_diag_tensor = torch.sum(abs_R, dim=1).to_dense()
    v_diag_tensor = torch.sum(abs_R, dim=0).to_dense()

    u_diag_tensor[u_diag_tensor == 0] = 1
    v_diag_tensor[v_diag_tensor == 0] = 1

    u_diag_inv_list = 1.0 / u_diag_tensor
    v_diag_int_list = 1.0 / v_diag_tensor

    return u_diag_inv_list, v_diag_int_list


def normalization(
    device: str,
    R_pos: torch.Tensor,
    R_neg: torch.Tensor,
    d_u_inv: torch.Tensor,
    d_v_inv: torch.Tensor
) -> torch.Tensor:
    """ normalization

    Args:
        R_pos (Tensor): bi-adjacency matrix related to positive link
        R_neg (Tensor): bi-adjacency matrix related to negative link
        d_u_inv (Tensor): degree matrix about node type "u" - row
        d_v_inv (Tensor): degree matrix about node type "v" - col

    Returns:
        A_pos (Tensor): normalized matrix of posivtve about u to v
        A_neg (Tensor): normalized matrix of negative about u to v
        B_pos (Tensor): normalized matrix of posivtve about v to u
        B_neg (Tensor): normalized matrix of negative about v to u
    """

    R_pos = R_pos.to(device)
    R_neg = R_neg.to(device)
    d_u_inv = d_u_inv.to(device)
    d_v_inv = d_v_inv.to(device)
    d_u_inv = torch.diag(d_u_inv)
    d_v_inv = torch.diag(d_v_inv)
    A_pos = torch.mm(d_u_inv, R_pos).to_sparse()
    A_neg = torch.mm(d_u_inv, R_neg).to_sparse()

    B_pos = torch.mm(d_v_inv, R_pos.t()).to_sparse()
    B_neg = torch.mm(d_v_inv, R_neg.t()).to_sparse()

    return A_pos.to("cpu"), A_neg.to("cpu"), B_pos.to("cpu"), B_neg.to("cpu")


def svd(
    device: str,
    normalized_pos_matrix: torch.tensor,
    normalized_neg_matrix: torch.tensor,
    rank: int
):
    """svd

    Args:
        M_pos (Tensor.sparse): pos adj matrix 
        M_neg (Tensor.sparse): neg adj matrix
        rank (int): SVD Rank

    Returns:
        (dict): stored dictionary about element of SVD\
            from two given(it stored to 2 cases(pos/neg) 6 elements) 
    """
    normalized_pos_matrix = normalized_pos_matrix.to(device)
    normalized_neg_matrix = normalized_neg_matrix.to(device)
    store_dict = dict()
    M = [normalized_pos_matrix, normalized_neg_matrix]
    
    key_usv_list = ['u', 's', 'v']
    key_sign_list = ['pos', 'neg']

    key_list = [key_usv_list[j] + '_' + key_sign_list[i]
                for i in range(len(key_sign_list))
                for j in range(len(key_usv_list))]

    for i in range(len(M)):
        U, S, V = torch.svd_lowrank(M[i].coalesce(), rank)

        store_dict[key_list[0+(3*i)]] = U.to("cpu")
        store_dict[key_list[1+(3*i)]] = S.to("cpu")
        store_dict[key_list[2+(3*i)]] = V.to("cpu")

    return store_dict



