import torch
from ELISE import elise
class Encoder(object):
    """encoder class for training model
    Args:
        **param: additional parameters
    """

    def __init__(self, dataset, **param):
        self.param = param
        self.model_name = param["model"]
        self.encoder_method = self.set_encoder_method(dataset)

    def update_embedding(self):
        """
        forward function for training model

        Args:
            x (torch.Tensor): input tensor

        Returns:
            embeddings (torch.Tensor): embeddings
        """
        self.encoder_method.train()
        embeddings = self.encoder_method()
        return embeddings

    def set_encoder_method(self, dataset):
        """
        set model function for training model

        Args:
            model (object): model object
        """
        encoder_method = elise(**self.param)
        encoder_method.build_structure(dataset)
        return encoder_method.to(self.param["device"])

    def get_encoder_method(self):
        """
        get model function for training model

        Returns:
            model (object): model object
        """
        return self.encoder_method

