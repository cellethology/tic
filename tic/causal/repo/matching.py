# matching.py

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from tic.causal.base import BaseCausalMethod
from tic.causal.causal_input import CausalInput


logger = logging.getLogger(__name__)


class MatchingCausalMethod(BaseCausalMethod):
    """
    Implements a simple nearest-neighbor (1-to-1) matching approach.
    """

    def __init__(self):
        super().__init__(name="Matching")
        self.matched_pairs_ = None   # Will store matched indices
        self.is_fit_ = False

    def fit(
        self,
        input_data: CausalInput,
        n_neighbors: int = 1,
        distance_metric: str = "euclidean",
        random_state: Optional[int] = None,
        *args,
        **kwargs
    ):
        """
        Perform 1-to-1 nearest neighbor matching on the input dataset.
        
        :param input_data: A CausalInput dataclass containing:
            - data (pd.DataFrame)
            - treatment_col (str)
            - outcome_col (str) [optional, not strictly needed for matching]
            - covariates (List[str]) used for matching
        :param n_neighbors: Number of neighbors to match for each treated unit (default 1).
                            In this example, we keep it at 1 for 1-to-1 matching.
        :param distance_metric: Metric passed to NearestNeighbors (e.g. "euclidean", "minkowski", etc.)
        :param random_state: For reproducible shuffling of control units, if desired.
        """
        try:
            df = input_data.data.copy()
            treat_col = input_data.treatment_col
            covariates = input_data.covariates

            logger.info(f"[Matching] Starting nearest-neighbor matching with {n_neighbors} neighbor(s).")

            if not covariates:
                raise ValueError("No covariates provided for matching.")

            # Split into treated vs. control
            treated_df = df[df[treat_col] == 1].copy()
            control_df = df[df[treat_col] == 0].copy()

            if treated_df.empty or control_df.empty:
                raise ValueError("Either treated or control group is empty; cannot perform matching.")

            # Extract covariate arrays
            X_treated = treated_df[covariates].values
            X_control = control_df[covariates].values

            # Fit NearestNeighbors on the control group
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric=distance_metric)
            nn.fit(X_control)

            # Find the nearest neighbors for each treated unit
            distances, indices = nn.kneighbors(X_treated)

            # If we want 1-to-1 matching (n_neighbors=1), 
            # 'indices' is of shape (num_treated, 1)
            # each row i is the index of the matched control row in X_control
            matched_indices_control = indices.flatten()

            # Optionally, we could exclude duplicates so each control is used at most once,
            # but for demonstration, let's keep it simple. If you need no replacement, 
            # you'd need more advanced logic to ensure unique matches.

            # Build a list of matched pairs (treated_index, control_index)
            treated_index_list = treated_df.index.tolist()
            matched_pairs = list(zip(treated_index_list, [control_df.index[i] for i in matched_indices_control]))

            self.matched_pairs_ = matched_pairs
            self.is_fit_ = True

            logger.info("[Matching] Matching completed. Found matches for all treated units.")
        except Exception as e:
            logger.error(f"[Matching] Error during fit: {e}")
            raise

    def estimate_effect(
        self,
        input_data: CausalInput,
        effect_type: str = "ATE",
        *args,
        **kwargs
    ) -> float:
        """
        Estimate the treatment effect using the matched pairs.
        
        :param input_data: CausalInput object with an outcome_col specified
        :param effect_type: "ATE", "ATT", or "ATC" (this example only demonstrates ATE).
        :return: Estimated effect (float).
        """
        if not self.is_fit_ or self.matched_pairs_ is None:
            raise RuntimeError("Cannot estimate effect: MatchingCausalMethod is not fit or no pairs found.")

        if input_data.outcome_col is None:
            raise ValueError("Outcome column must be specified in CausalInput to estimate effect.")

        # We'll define the ATE as the average difference in outcome between matched pairs.
        # ATE ~ (1/N) * sum( outcome_treated - outcome_control )
        try:
            df = input_data.data
            outcome_col = input_data.outcome_col

            # Collect differences for each matched pair
            differences = []
            for (i_treated, i_control) in self.matched_pairs_:
                y_treated = df.loc[i_treated, outcome_col]
                y_control = df.loc[i_control, outcome_col]
                differences.append(y_treated - y_control)

            effect_estimate = np.mean(differences)

            logger.info(f"[Matching] Estimated {effect_type}: {effect_estimate:.4f}")
            return effect_estimate
        except Exception as e:
            logger.error(f"[Matching] Error during estimate_effect: {e}")
            raise
    
def main():
    np.random.seed(42)
    data_size = 100

    df = pd.DataFrame({
        "treated": np.random.binomial(1, p=0.4, size=data_size),
        "cov1": np.random.randn(data_size),
        "cov2": np.random.randn(data_size),
        "outcome": np.random.randn(data_size) + 2  # shift for demonstration
    })

    # Build the input object
    c_input = CausalInput(
        data=df,
        treatment_col="treated",
        outcome_col="outcome",
        covariates=["cov1", "cov2"]
    )

    # Instantiate the MatchingCausalMethod
    matcher = MatchingCausalMethod()

    # Fit the matching model (1-nearest neighbor, default Euclidean)
    matcher.fit(
        input_data=c_input,
        n_neighbors=1,
        distance_metric="euclidean",
        random_state=42
    )

    # Estimate effect (ATE)
    effect = matcher.estimate_effect(c_input, effect_type="ATE")
    print("Estimated Matching Effect:", effect)

if __name__ == "__main__":
    main()