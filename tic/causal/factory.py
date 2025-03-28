# tic/causal/factory.py

from tic.causal.repo.difference_in_diff import DifferenceInDifferencesMethod
from tic.causal.repo.granger_causality import GrangerCausalityMethod
from tic.causal.repo.instrumental_vars import InstrumentalVariableMethod
from tic.causal.repo.matching import MatchingCausalMethod
from tic.causal.repo.propensity_score import PropensityScoreMethod


class CausalMethodFactory:
    """
    A simple factory that returns the correct causal method object
    based on a string identifier.
    """
    @staticmethod
    def get_method(method_name: str):
        if method_name.lower() == "matching":
            return MatchingCausalMethod()
        elif method_name.lower() == "propensityscore":
            return PropensityScoreMethod()
        elif method_name.lower() == "instrumentalvariables":
            return InstrumentalVariableMethod()
        elif method_name.lower() == "differenceindifferences":
            return DifferenceInDifferencesMethod()
        elif method_name.lower() == "granger_causality":
            return GrangerCausalityMethod()
        else:
            raise ValueError(f"Unknown causal method: {method_name}")