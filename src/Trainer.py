
import torch

from tqdm import tqdm
from loguru import logger
from utils import metric
from Encoder import Encoder
from Decoder import Decoder
from itertools import chain


class Trainer(object):
    """trainer class for training model

    Args:
        pre_processed_data (dict): pre_processed_data from dataloader
        **param: additional parameters
    """

    def __init__(self, pre_processed_data, **param):
        self.param = param
        self.pre_processed_data = pre_processed_data
        self.encoder = Encoder(pre_processed_data, **param)  # GNN model
        self.decoder = Decoder(**param)  # decoder_class
        self.optim = self.set_optimizer()

    def set_optimizer(self):
        """
        optim setting function for training model

        Returns:
            optim (object): optim object
        """

        optims = {"adam": torch.optim.Adam}
        if self.param["optimizer"].lower() in optims:
            return optims[self.param["optimizer"].lower()](
                chain(
                    self.encoder.get_encoder_method().parameters(),
                    self.decoder.get_link_sign_classifier().parameters()
                ),
                lr=self.param["lr"],
                weight_decay=self.param["wdc"]
            )
        else:
            raise "not supported optimizer"

    def encoding(self):  # == model
        """
        encoder function for training model

        Args:
            model (object): model object

        Returns:
            embeddings (torch.Tensor): embeddings
        """
        embedding = self.encoder.update_embedding()
        return embedding

    def decoding(self, embeddings, preprocessed_data):  # == loss calculation
        """
        decoder function for training model

        Args:
            decoder (object): decoder object
            embeddings (torch.Tensor): embeddings
        """
        loss, sign_loss = self.decoder.calculate_loss(
            embeddings, preprocessed_data)
        return loss, sign_loss

    def train(self):
        """
        train function for training model
        """
        
        result_list = []
        for epoch in tqdm(range(self.param["epochs"])):
            self.optim.zero_grad()
            embeddings = self.encoding()
            
            loss, sign_loss = self.decoding(embeddings, self.pre_processed_data)
            loss.backward()
            
            self.optim.step()
            
            result = self.evaluate(embeddings, self.pre_processed_data)
            result["train_loss"] = {
                "loss_sum" : loss,
                "sign_loss" : sign_loss,
            }
            result_list.append(result)
            
        
        return result_list

    def evaluate(self, embeddings, dataset):
        """_summary_

        Args:
            embeddings (torch.tensor): the embedding from encoder
            dataset (dict): mapped dataset

        Returns:
            metrics: calculated each metrics after through decoder
        """
        # codes for experiments
        # run with torch.no_grad() and model.eval()
        with torch.no_grad():
            train_prob = self.decoder.sign_predict(
                embeddings, dataset["train_edges"], True)
            train_metric = metric(dataset["train_label"], train_prob)
            
            val_prob = self.decoder.sign_predict(
                embeddings, dataset["valid_edges"], True)
            val_metric = metric(dataset["valid_label"], val_prob)
            
            test_prob = self.decoder.sign_predict(
                embeddings, dataset["test_edges"], True)
            test_metric = metric(dataset["test_label"], test_prob)
        return {"train": train_metric, "valid": val_metric, "test": test_metric}

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, loss):
        if self.best_loss is None:
            self.best_loss = loss
        
        elif loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        else:
            self.best_loss = loss
            self.counter = 0