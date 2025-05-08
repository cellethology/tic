# tic/wrappers/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Generic, TypeVar

T_cfg = TypeVar("T_cfg")       # concrete config type (e.g. PseudotimeConfig)
T_out = TypeVar("T_out")       # concrete return type (e.g. AnnData)


class BaseWrapper(ABC, Generic[T_cfg, T_out]):
    """Shared helper - cache result & expose a tiny public surface."""

    def __init__(self, cfg: T_cfg) -> None:
        self.cfg = cfg
        self._result: T_out | None = None

    # --------------------------------------------------------------------- API
    def run(self, *args, **kwargs) -> T_out:
        return self.fit(*args, **kwargs)

    def fit(self, *args, **kwargs) -> T_out:  # noqa: D401
        self._result = self._fit_impl(*args, **kwargs)
        return self._result

    @property
    def result(self) -> T_out:
        if self._result is None:  # pragma: no cover
            raise RuntimeError("Run `.fit()` before accessing results.")
        return self._result

    # ------------------------------------------------------------------ hooks
    @abstractmethod
    def _fit_impl(self, *args, **kwargs) -> T_out:  # noqa: D401
        """Internal hook executed by `.fit()`."""

    def save_params(self, path: Path | str):
        """
        Save the parameters of the wrapper to a file as json format.
        """
        with open(path, 'w') as f:
            json.dump(self.cfg.to_dict(), f)
