# base.py

import logging
from abc import ABC, abstractmethod

from core.causal.causal_input import CausalInput

logger = logging.getLogger(__name__)

class BaseCausalMethod(ABC):
    """
    Abstract base class that defines the interface for 
    all causal methods in the repo.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, input_data: CausalInput, *args, **kwargs):
        """
        Prepare or train the method on the dataset.
        """
        pass

    @abstractmethod
    def estimate_effect(self, input_data: CausalInput, *args, **kwargs):
        """
        After fitting, estimate the treatment effect (or parameters).
        """
        pass