from abc import ABC
from abc import abstractmethod
from enum import Enum


class CodeBehavior(Enum):

    HPT = "Hyper-parameter tuning"
    MC = "Best model comparing"
    MR = "Best model comparing (just running)"


class IBehavior(ABC):

    @abstractmethod
    def model_search_mode(self):
        pass

    @abstractmethod
    def save_best_search(self):
        pass

    @abstractmethod
    def save_best_comparison(self):
        pass


class HyperParameterTuning(IBehavior):

    def model_search_mode(self):
        return True

    def save_best_search(self):
        return True

    def save_best_comparison(self):
        return False


class ModelsComparing(IBehavior):

    def model_search_mode(self):
        return False

    def save_best_search(self):
        return False

    def save_best_comparison(self):
        return True


class ModelRunning(IBehavior):

    def model_search_mode(self):
        return False

    def save_best_search(self):
        return False

    def save_best_comparison(self):
        return False
