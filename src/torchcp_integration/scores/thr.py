import torch
from base import BaseScore


class THR(BaseScore):
    """
    Threshold conformal predictors 
    
    Args:
        score_type (Union[str, Callable], optional): Specifies how to transform logits.
            - If str: Use predefined functions {"softmax", "identity", "log_softmax", "log"}
            - If callable: Custom function that takes and returns torch.Tensor
            Defaults to "softmax".
    """

    def __init__(self, score_type="softmax"):
        super().__init__()

        self.score_type = score_type

        if callable(score_type):
            self.transform = score_type
        else:
            if score_type == "identity":
                self.transform = lambda x: x
            elif score_type == "softmax":
                self.transform = lambda x: torch.softmax(x, dim=-1)
            elif score_type == "log_softmax":
                self.transform = lambda x: torch.log_softmax(x, dim=-1)
            elif score_type == "log":
                self.transform = lambda x: torch.log(x)
            else:
                raise ValueError(
                    f"Score type '{score_type}' is not implemented. Options are 'softmax', 'identity', 'log_softmax', 'log', or a callable function.")

    def __call__(self, logits, label=None):

        if len(logits.shape) > 2:
            raise ValueError("dimension of logits are at most 2.")

        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = self.transform(logits)
        if label is None:
            return self._calculate_all_label(probs)
        else:
            return self._calculate_single_label(probs, label)

    def _calculate_single_label(self, probs, label):
        """
            Calculate non-conformity score for a single label.
        """
        return 1 - probs[torch.arange(probs.shape[0], device=probs.device), label]

    def _calculate_all_label(self, probs):
        """
            Calculate non-conformity scores for all labels.
        """
        return 1 - probs