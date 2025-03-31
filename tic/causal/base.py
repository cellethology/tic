"""
Module: tic.causal.base

Defines the abstract base class for causal inference methods.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from tic.causal.causal_input import CausalInput

logger = logging.getLogger(__name__)


class BaseCausalMethod(ABC):
    """
    Abstract base class that defines the interface for all causal methods.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the causal method.

        Parameters
        ----------
        name : str
            The name identifier for the causal method.
        """
        self.name = name

    @abstractmethod
    def fit(self, input_data: CausalInput, *args, **kwargs) -> None:
        """
        Prepare or train the causal method on the provided dataset.

        Parameters
        ----------
        input_data : CausalInput
            The causal input data.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        """
        pass

    @abstractmethod
    def estimate_effect(self, input_data: CausalInput, *args, **kwargs) -> Any:
        """
        After fitting, estimate the treatment effect or other relevant parameters.

        Parameters
        ----------
        input_data : CausalInput
            The causal input data.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        any
            The estimated causal effect or parameters.
        """
        pass