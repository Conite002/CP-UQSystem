import warnings
from abc import ABCMeta, abstractmethod

import torch

from utils.Conf_calibration import ConfCalibrator
from metrics.metrics import Metrics
from utils.common import get_device


class BasePredictor(object):
    """
    Abstract base class for all conformal predictors.
        
    Args:
        score_function (callable): Non-conformity score function.
        model (torch.nn.Module, optional): A PyTorch model. Default is None.
        temperature (float, optional): The temperature of Temperature Scaling. Default is 1.
    
    Attributes:
        score_function (callable): Non-conformity score function.
        _model (torch.nn.Module): The PyTorch model.
        _device (torch.device): The device on which the model is located.
        _metric (Metrics): An instance of the Metrics class.
        _logits_transformation (ConfCalibrator): The logits transformation using Temperature Scaling.
        
    Methods:
        calibrate(cal_dataloader, alpha):
            Virtual method to calibrate the calibration set.
        predict(x_batch):
            Generate prediction sets for the test examples.
        _generate_prediction_set(scores, q_hat):
            Generate the prediction set with the threshold q_hat.
    """

    __metaclass__ = ABCMeta

    def __init__(self, score_function, model=None, temperature=1):

        warnings.warn(
            "The 'temperature' parameter is deprecated and will be removed in a future version. "
            "Use torchcp.classification.traienr.TemperatureScalingModel instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0.")

        # self.score_function = score_function
        # self._model = model
        # if self._model != None and self:
        #     self._model.eval()
        # self._device = get_device(model)
        # self._metric = Metrics()
        # self._logits_transformation = ConfCalibrator.registry_ConfCalibrator("TS")(temperature).to(self._device)

        
        self.score_function = score_function
        self._model = model
        self.is_ensemble = isinstance(model, list)  
        self._device = get_device(model)

        if self._model is not None:
            if self.is_ensemble:
                print(f"[INFO] Using an ensemble of {len(model)} models.")
                for m in self._model:
                    m.eval().to(self._device)
            else:
                self._model.eval().to(self._device)

        self._metric = Metrics()
        self._logits_transformation = ConfCalibrator.registry_ConfCalibrator("TS")(temperature).to(self._device)


    
    @abstractmethod
    def calibrate(self, cal_dataloader, alpha):
        """
        Virtual method to calibrate the calibration set.

        Args:
            cal_dataloader (torch.utils.data.DataLoader): A dataloader of the calibration set.
            alpha (float): The significance level.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_batch):
        """
        Generate prediction sets for the test examples.
        
        Args:
            x_batch (torch.Tensor): A batch of input.
        """
        raise NotImplementedError

    def _generate_prediction_set(self, scores, q_hat: torch.Tensor):
        """
        Generate the prediction set with the threshold q_hat.

        Args:
            scores (torch.Tensor): The non-conformity scores of {(x,y_1),..., (x,y_K)}.
            q_hat (torch.Tensor): The calibrated threshold.

        Returns:
            torch.Tensor: A tensor of 0/1 values indicating the prediction set for each example.
        """

        return (scores <= q_hat).int()

    def get_device(self):
        return self._device

    def _get_logits(self, x_batch):
        """
        Returns model logits for a given input batch.

        Args:
            x_batch (torch.Tensor): A batch of input images.

        Returns:
            torch.Tensor: Model logits.
        """
        x_batch = x_batch.to(self._device)

        if self.is_ensemble:
            logits = torch.stack([model(x_batch) for model in self._model]).mean(dim=0) 
        else:
            logits = self._model(x_batch)

        return logits