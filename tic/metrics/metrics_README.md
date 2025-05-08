# `tic.metrics` – Time-Series Analysis Module

The `tic.metrics` submodule provides a unified interface for trend detection and monotonicity analysis of time series data. It supports individual and batch processing of biological signals or other sequential measurements, such as gene expression over pseudotime.

---

## 📦 Core API

```python
from tic.metrics import (
    calculate_monotonicity,
    calculate_trend,
    rank_by_monotonicity,
    rank_by_trend
)
```

### ✅ `calculate_monotonicity`

Compute the monotonicity (rank correlation) between a time series and time.

```python
calculate_monotonicity(y: np.ndarray, x: Optional[np.ndarray], method="spearman")
```

- Methods:
  - `"spearman"` – Spearman’s rank correlation.
  - `"kendall"` – Kendall's tau.
- Returns: `{ "method": str, "correlation": float, "p_value": float }`

---

### ✅ `calculate_trend`

Detect trends in the time series using statistical models.

```python
calculate_trend(y: np.ndarray, x: Optional[np.ndarray], method="mann_kendall")
```

- Methods:
  - `"mann_kendall"` – Non-parametric test for monotonic trends.
  - `"linear_regression"` – Fit and evaluate a linear regression model.
- Returns: a dictionary of method-specific results.

---

### 🔢 `rank_by_monotonicity` / `rank_by_trend`

Rank a list of time series according to their monotonicity or trend strength.

```python
rank_by_monotonicity([series1, series2, ...], method="kendall")
rank_by_trend([series1, series2, ...], method="mann_kendall")
```

- Returns: list of indices (optionally with scores).

---

## 🧠 Internals

### `monotonicity.py`
Implements:
- `spearman_monotonicity`
- `kendall_monotonicity`

### `trend.py`
Implements:
- `mann_kendall_trend`
- `linear_regression_trend`

Both modules use `_prepare_data` to align `x` and `y` input arrays.

---

## 📊 Usage Example

```python
import numpy as np
from tic.metrics import calculate_monotonicity, calculate_trend

# Fake biomarker expression over time
y = np.array([1.2, 1.5, 1.7, 2.0, 2.1])
x = np.arange(len(y))

# Monotonicity
mon = calculate_monotonicity(y, x, method="spearman")

# Trend
trend = calculate_trend(y, x, method="linear_regression")

print(mon)
print(trend)
```

---

## 📁 Structure

- `api.py` – Unified entry point
- `monotonicity.py` – Rank-based monotonicity metrics
- `trend.py` – Regression and non-parametric trend metrics

---

## 🔬 Intended Use

This module is primarily used in:
- Ranking biomarkers along pseudotime
- Measuring temporal smoothness or directional shifts
- Screening for time-dependent biological markers

