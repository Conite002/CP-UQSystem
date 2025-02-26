from abc import ABC, abstractmethod

class BaseScore(ABC):
    """
    Abstract base class for all score functions.
    """
    # __metaclass__ = ABCMeta

    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, logits, labels=None):
        """Virtual method to compute scores for a data pair (x,y).

        Args:
            probs (torch.Tensor): The prediction probabilities.
            label (torch.Tensor): The ground truth label.
        """
        raise NotImplementedError