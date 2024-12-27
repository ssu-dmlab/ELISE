# decoder
import torch.nn as nn
import torch


class Decoder(object):
    def __init__(self, **param):
        
        self.param = param
        
        if param["num_decoder_layers"] != 0:
            self.LinkSignClassifier = LinkSignClassifier(
                self.param["decoder_input_dim"], 1, self.param["num_decoder_layers"]).to(param["device"])
        else:
            self.LinkSignClassifier = DotProductDecoder().to(param["device"])
        self.bceloss = torch.nn.BCELoss()

    def sign_predict(self, embeddings, edges, eval=False):
        """_summary_

        Args:
            embeddings (object): the embeddings from encoder
            edges (torch.tensor): the edge list
            eval (bool): if training phase then set to false, otherwise, set to true. Defaults to False.

        Raises:
            ValueError: it occur by param["node_idx_type"]

        Returns:
            logit (torch.tensor): the logit value after through classifier
        """
        
        #train or eval
        if eval:
            self.LinkSignClassifier.eval()
        else:
            self.LinkSignClassifier.train()
            
        #emb type check
        if self.param["node_idx_type"] == "bi":
            user_emb, item_emb = embeddings
            src_features = user_emb[edges[:, 0], :] #user
            dst_features = item_emb[edges[:, 1], :] #item
        
        elif self.param["node_idx_type"] == "uni":
            src_features = embeddings[edges[:, 0], :]
            dst_features = embeddings[edges[:, 1], :]
        
        else:
            raise ValueError
        
        features = torch.cat((src_features, dst_features), dim=1)
        logit = self.LinkSignClassifier(features).squeeze()
        return logit

    def calculate_loss(self, embeddings, preprocessed_data):
        """
        This method is decoder for train the model

        Args:
            embeddings (torch.Tensor): embeddings
            preprocessed_data (dict): edges
            y (torch.Tensor): labels
        """
        train_edges, train_label = preprocessed_data["train_edges"], preprocessed_data["train_label"].float()
        train_label[train_label == -1] = 0
        
        # sign loss
        sign_prob = self.sign_predict(embeddings, train_edges)
        assert train_label.shape[0] == sign_prob.shape[0]
        loss = self.bceloss(sign_prob, train_label)
        
        sign_loss = loss.item()
        return loss, sign_loss

    def get_link_sign_classifier(self):
        """
        This method import the classifier for train the model

        Args:
            embeddings (torch.Tensor): embeddings
        """
        return self.LinkSignClassifier

class DotProductDecoder(nn.Module):
    """
    Decoder for link sign prediction.
    """
    def __init__(self):
        super(DotProductDecoder, self).__init__()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, edge_embeddings):
        embedding_a_edges = edge_embeddings[:, :edge_embeddings.size(1) // 2]
        embedding_b_edges = edge_embeddings[:, edge_embeddings.size(1) // 2:]
        elementwise_mul = embedding_a_edges * embedding_b_edges  # (N, D)
        y = elementwise_mul.sum(dim=1)  # (N,)
        return self.sigmoid(y)
    
class LinkSignClassifier(nn.Module):
    """
    This method is consist of classifier by designated parameters

    Args:
        in_dim(int): input dimension
        out_dim(int): hidden dimension
        num_layers(int): number of layers
    """

    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        if self.out_dim != 1:
            raise ValueError("Error: out_dim should be 1!")

        modules = []
        for i in range(self.num_layers-1):
            if i == 0:
                modules.append(nn.Linear(self.in_dim, self.in_dim//2))
            else:
                modules.append(nn.ReLU())
                modules.append(nn.Linear(self.in_dim//2, self.in_dim//2))

        if self.num_layers == 1:
            modules.append(nn.Linear(self.in_dim, self.out_dim))
        else:
            modules.append(nn.ReLU())
            modules.append(nn.Linear(self.in_dim//2, self.out_dim))
        modules.append(nn.Sigmoid())

        self.classifier = nn.Sequential(*modules)

    def forward(
        self,
        edge_embeddings: torch.Tensor
    ) -> torch.Tensor:
        return self.classifier(edge_embeddings)
