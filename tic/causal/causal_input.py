"""
Module: tic.causal.causal_input

Encapsulates the data and column specifications required by most causal inference methods.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd


@dataclass
class CausalInput:
    """
    Encapsulates the data and column specifications needed by causal inference methods.

    Attributes
    ----------
    data : pd.DataFrame
        The dataset as a pandas DataFrame.
    treatment_col : str
        The column name representing the treatment variable.
    outcome_col : Optional[str]
        The column name representing the outcome variable.
    covariates : List[str]
        List of covariate column names.
    extra_params : Dict[str, Any]
        Additional parameters (e.g., instruments, group_col, etc.).
    """
    data: pd.DataFrame
    treatment_col: str
    outcome_col: Optional[str] = None
    covariates: List[str] = field(default_factory=list)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Basic validation checks.
        if self.treatment_col not in self.data.columns:
            raise ValueError(f"Treatment column '{self.treatment_col}' not in DataFrame.")
        if self.outcome_col and self.outcome_col not in self.data.columns:
            raise ValueError(f"Outcome column '{self.outcome_col}' not in DataFrame.")
        for cov in self.covariates:
            if cov not in self.data.columns:
                raise ValueError(f"Covariate '{cov}' not in DataFrame.")