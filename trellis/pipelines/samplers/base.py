from typing import *
from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    A base class for samplers.
    """

    @abstractmethod
    def sample(
        self,
        model,
        **kwargs
    ):
        """
        Sample from a model.
        """
        pass

    def repaint(
        self,
        model,
        **kwargs
    ):
        """
        Repaint with a model.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement repaint().")

    def sdedit(
        self,
        model,
        **kwargs
    ):
        """
        SDEdit with a model.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement sdedit().")
    
