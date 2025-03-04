from core.causal.repo.difference_in_diff import DifferenceInDifferencesMethod
from core.causal.repo.instrumental_vars import InstrumentalVariableMethod
from core.causal.repo.matching import MatchingCausalMethod
from core.causal.repo.propensity_score import PropensityScoreMethod
from core.causal.repo.regression_discontinuity import RegressionDiscontinuityMethod
from core.causal.repo.sem import StructuralEquationModelMethod


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
        elif method_name.lower() == "regressiondiscontinuity":
            return RegressionDiscontinuityMethod()
        elif method_name.lower() == "differenceindifferences":
            return DifferenceInDifferencesMethod()
        elif method_name.lower() == "sem":
            return StructuralEquationModelMethod()
        else:
            raise ValueError(f"Unknown causal method: {method_name}")