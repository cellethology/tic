# causal_input.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import pandas as pd

@dataclass
class CausalInput:
    """
    Encapsulates the data and column specifications needed 
    by most causal inference methods.
    """
    data: pd.DataFrame
    treatment_col: str
    outcome_col: Optional[str] = None
    covariates: List[str] = field(default_factory=list)
    
    # If you need extra parameters (e.g., instruments, group_col, etc.),
    # you can either define more fields or store them in a dictionary:
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Basic validation checks
        if self.treatment_col not in self.data.columns:
            raise ValueError(f"Treatment column '{self.treatment_col}' not in DataFrame.")
        if self.outcome_col and self.outcome_col not in self.data.columns:
            raise ValueError(f"Outcome column '{self.outcome_col}' not in DataFrame.")
        for cov in self.covariates:
            if cov not in self.data.columns:
                raise ValueError(f"Covariate '{cov}' not in DataFrame.")