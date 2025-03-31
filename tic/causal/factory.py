"""
Module: tic.causal.factory

Provides a factory to instantiate causal method objects based on a string identifier.
"""

from tic.causal.repo.granger_causality import GrangerCausalityMethod


class CausalMethodFactory:
    """
    Factory class for creating causal method objects.
    """

    @staticmethod
    def get_method(method_name: str):
        """
        Return the causal method instance corresponding to the given identifier.

        Parameters
        ----------
        method_name : str
            The identifier for the causal method (e.g., "granger_causality").

        Returns
        -------
        An instance of a causal method.

        Raises
        ------
        ValueError
            If the method name is unknown.
        """
        if method_name.lower() == "granger_causality":
            return GrangerCausalityMethod()
        else:
            raise ValueError(f"Unknown causal method: {method_name}")