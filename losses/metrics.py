from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class Metric(ABC):
    @abstractmethod
    def refresh(self):
        pass

    @abstractmethod
    def update(self, predictions, targets):
        pass

    @abstractmethod
    def calculate(self):
        pass


class Accuracy(Metric):
    def __init__(self):
        self._num_matches = 0
        self._total_num = 0

    def refresh(self):
        self._num_matches = 0
        self._total_num = 0

    def update(self, predictions, targets):
        self._total_num += len(targets)
        matches = (np.argmax(predictions[:, :, 0, 0], axis=1) == targets[:, 0, 0]).sum()
        self._num_matches += matches

    def calculate(self):
        if self._total_num == 0:
            return 0
        else:
            return self._num_matches / self._total_num


class F1Score(Metric):
    def __init__(self):
        self._predictions = []
        self._targets = []

    def refresh(self):
        self._predictions = []
        self._targets = []

    def update(self, predictions, targets):
        self._predictions.append(np.argmax(predictions[:, :, 0, 0], axis=1))
        self._targets.append(targets[:, 0, 0])

    def calculate(self):
        predictions = np.concatenate(self._predictions)
        targets = np.concatenate(self._targets)
        prs, rcls, f1s, sups = precision_recall_fscore_support(targets, predictions)
        return f1s.mean()
