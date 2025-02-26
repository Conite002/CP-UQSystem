from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from base import BasePredictor
from utils.common import calculate_conformal_value


class SplitPredictor_CM(BasePredictor):
    """
    Split Conformal Prediction (Vovk et a., 2005).
    Book: https://link.springer.com/book/10.1007/978-3-031-06649-8.
    
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module or List[torch.nn.Module], optional): A PyTorch model or an ensemble of models.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
        use_mc_dropout (bool, optional): If True, enables Monte Carlo Dropout during inference.
        mc_samples (int, optional): Number of stochastic forward passes for MC-Dropout.
    """

    def __init__(self, score_function, model=None, temperature=1, use_mc_dropout=False, mc_samples=10):
        super().__init__(score_function, model, temperature)
        self.use_mc_dropout = use_mc_dropout
        self.mc_samples = mc_samples
        
        self.if_ensemble = isinstance(model, list)
        if self.if_ensemble:
            print(f"[INFO] Using Ensemble Learning with {len(model)} models.")


    def calibrate(self, cal_dataloader, alpha):
        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for examples in cal_dataloader:
                tmp_x, tmp_labels = examples[0].to(self._device), examples[1].to(self._device)
                tmp_logits = self._get_logits(tmp_x)
                logits_list.append(tmp_logits)
                labels_list.append(tmp_labels)
            logits = torch.cat(logits_list).float()
            labels = torch.cat(labels_list)
        self.calculate_threshold(logits, labels, alpha)

    def calculate_threshold(self, logits, labels, alpha):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels)
        self.q_hat = self._calculate_conformal_value(scores, alpha)

    def _calculate_conformal_value(self, scores, alpha):
        return calculate_conformal_value(scores, alpha)

    #############################
    # The prediction process
    ############################
    def predict(self, x_batch):
        """
        Generate prediction sets for a batch of instances.

        Args:
            x_batch (torch.Tensor): A batch of instances.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """

        if self._model is None:
            raise ValueError("Model is not defined. Please provide a valid model.")

        self._model.eval()
        x_batch = self._get_logits(x_batch)
        x_batch = self._logits_transformation(x_batch).detach()
        sets = self.predict_with_logits(x_batch)
        return sets

    def predict_with_logits(self, logits, q_hat=None):
        """
        Generate prediction sets from logits.

        Args:
            logits (torch.Tensor): Model output before softmax.
            q_hat (torch.Tensor, optional): The conformal threshold. Default is None.

        Returns:
            list: A list of prediction sets for each instance in the batch.
        """

        scores = self.score_function(logits).to(self._device)
        if q_hat is None:
            if self.q_hat is None:
                raise ValueError("Ensure self.q_hat is not None. Please perform calibration first.")
            q_hat = self.q_hat

        S = self._generate_prediction_set(scores, q_hat)

        return S

    #############################
    # The evaluation process
    ############################

    def evaluate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate prediction sets on validation dataset.
        
        Args:
            val_dataloader (torch.utils.data.DataLoader): Dataloader for validation set.
        
        Returns:
            dict: Dictionary containing evaluation metrics:
                - Coverage_rate: Empirical coverage rate on validation set
                - Average_size: Average size of prediction sets
        """
        predictions_sets_list: List[torch.Tensor] = []
        labels_list: List[torch.Tensor] = []

        # Evaluate in inference mode
        self._model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                # Move batch to device and get predictions
                inputs = batch[0].to(self._device)
                labels = batch[1].to(self._device)

                # Get predictions as bool tensor (N x C)
                batch_predictions = self.predict(inputs)

                # Accumulate predictions and labels
                predictions_sets_list.append(batch_predictions)
                labels_list.append(labels)

        # Concatenate all batches
        val_prediction_sets = torch.cat(predictions_sets_list, dim=0)  # (N_val x C)
        val_labels = torch.cat(labels_list, dim=0)  # (N_val,)

        # Compute evaluation metrics
        metrics = {
            "coverage_rate": self._metric('coverage_rate')(val_prediction_sets, val_labels),
            "average_size": self._metric('average_size')(val_prediction_sets, val_labels)
        }

        return metrics
    
    def _get_logits(self, x_batch):
        if self.use_mc_dropout:
            return self._mc_dropout_forward(x_batch)
        if self.is_ensemble:
            return self._ensemble_forward(x_batch)
        return self._model(x_batch.to(self._device)).float()
    
    def _mc_dropout_forward(self, x_batch):
        self._model.train()
        mc_logits = []
        for _ in range(self.mc_samples):
            with torch.no_grad():
                mc_logits.append(self._model(x_batch.to(self._device)).float())
        self._model.eval()
        return torch.stack(mc_logits).mean(dim=0)
    
    def _ensemble_forward(self, x_batch):
        ensemble_logits = []
        for model in self._model:
            with torch.no_grad():
                ensemble_logits.append(model(x_batch.to(self._device)).float())
        return torch.stack(ensemble_logits).mean(dim=0)
