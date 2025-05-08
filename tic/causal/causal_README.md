# `tic.causal`

The `tic.causal` module provides a general-purpose framework for **causal inference** over temporal or pseudo-temporal data. It supports the full causal inference pipeline, including data preparation, model abstraction, method registration, and specific algorithm implementations such as **Granger Causality**.

---

## 📦 Module Structure

```bash
tic/causal/
├── base.py              # Abstract base class for all causal methods
├── causal_input.py      # Standard data container for causal methods
├── factory.py           # Method dispatcher (string name → method instance)
├── repo/                # Concrete implementations (Granger)
```

---

## 🧩 Abstract Interfaces

### `BaseCausalMethod`

Every causal algorithm must implement:

```python
class BaseCausalMethod:
    def fit(self, input_data: CausalInput): ...
    def estimate_effect(self, input_data: CausalInput) -> Any: ...
```

This ensures compatibility and easy integration across the package.

---

## 📦 Input: `CausalInput`

Encapsulates:
- raw data (`data: pd.DataFrame`)
- required columns (`treatment_col`, `outcome_col`, `covariates`)
- optional metadata (`extra_params`, e.g. instruments, grouping)

```python
from tic.causal import CausalInput

ci = CausalInput(
    data=df,
    treatment_col="X",
    outcome_col="Y",
    covariates=["Z1", "Z2"]
)
```

---

## 🔨 Method Selection

Use the factory to get a causal method:

```python
from tic.causal.factory import CausalMethodFactory

method = CausalMethodFactory.get_method("granger_causality", maxlag=3)
method.fit(ci)
result = method.estimate_effect(ci)
```

---

## 🧪 Example: Granger Causality

```python
from tic.causal.repo.granger_causality import GrangerCausalityMethod

gc = GrangerCausalityMethod(maxlag=3)
gc.fit(ci)
result = gc.estimate_effect(ci)
```

Result includes:
- `p_value`, `best_lag`, `adjusted_pvalues`, `coefficients`, etc.

---