# SPDX-License-Identifier: MIT
"""
Model loader — thread-safe singleton for XGBoost model artefacts.

Loads once on first access; subsequent calls return the cached objects.
No Streamlit dependency (no ``@st.cache_resource``).

Usage::

    store = ModelStore.get()
    store.energy_model.predict(X_energy)
    store.temp_model.predict(X_temp)
"""

from __future__ import annotations

import pickle
import threading
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from core.config import (
    ENERGY_MODEL_FEATURES,
    ENERGY_MODEL_PKL,
    FEATURE_LIST_PKL,
    TEMP_MODEL_FEATURES,
    TEMP_MODEL_PKL,
)


class ModelLoadError(Exception):
    """Raised when a required model artefact cannot be loaded or validated."""


@dataclass
class ModelStore:
    """Immutable container holding the two trained XGBoost models and their
    per-model feature orderings.

    Obtain the singleton via :meth:`get`.  The first call loads from disk;
    all subsequent calls return the same instance.
    """

    energy_model: object  # xgboost.XGBRegressor
    temp_model: object    # xgboost.XGBRegressor
    energy_feature_names: List[str] = field(default_factory=list)
    temp_feature_names: List[str] = field(default_factory=list)
    feature_list_pkl: List[str] = field(default_factory=list)

    # ---- hyperparameter metadata (informational) ----
    energy_n_estimators: Optional[int] = None
    energy_learning_rate: Optional[float] = None
    energy_max_depth: Optional[int] = None
    temp_n_estimators: Optional[int] = None
    temp_learning_rate: Optional[float] = None
    temp_max_depth: Optional[int] = None

    # -- singleton plumbing --
    _instance: Optional["ModelStore"] = None
    _lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public singleton accessor
    # ------------------------------------------------------------------
    @classmethod
    def get(
        cls,
        energy_pkl: Optional[Path] = None,
        temp_pkl: Optional[Path] = None,
        feature_list_pkl_path: Optional[Path] = None,
    ) -> "ModelStore":
        """Return the singleton ``ModelStore``, loading models on the first
        call.

        Parameters are only used on the *first* invocation; subsequent
        calls ignore them and return the cached instance.
        """
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:          # double-checked locking
                return cls._instance
            cls._instance = cls._load(
                energy_pkl or ENERGY_MODEL_PKL,
                temp_pkl or TEMP_MODEL_PKL,
                feature_list_pkl_path or FEATURE_LIST_PKL,
            )
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached singleton (useful for tests)."""
        with cls._lock:
            cls._instance = None

    # ------------------------------------------------------------------
    # Internal: load + validate
    # ------------------------------------------------------------------
    @classmethod
    def _load(
        cls,
        energy_path: Path,
        temp_path: Path,
        feature_list_path: Path,
    ) -> "ModelStore":
        # --- Load energy model ---
        if not energy_path.exists():
            raise ModelLoadError(f"Energy model not found: {energy_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with open(energy_path, "rb") as f:
                energy_model = pickle.load(f)

        # --- Load temp model ---
        if not temp_path.exists():
            raise ModelLoadError(f"Temperature model not found: {temp_path}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with open(temp_path, "rb") as f:
                temp_model = pickle.load(f)

        # --- Load feature list pkl (informational) ---
        fl: List[str] = []
        if feature_list_path.exists():
            with open(feature_list_path, "rb") as f:
                fl = pickle.load(f)

        # --- Extract per-model feature orderings ---
        e_feats_raw = getattr(energy_model, "feature_names_in_", None)
        t_feats_raw = getattr(temp_model, "feature_names_in_", None)

        if e_feats_raw is None:
            raise ModelLoadError(
                "Energy model has no `feature_names_in_` attribute. "
                "Cannot determine expected feature ordering."
            )
        if t_feats_raw is None:
            raise ModelLoadError(
                "Temperature model has no `feature_names_in_` attribute. "
                "Cannot determine expected feature ordering."
            )

        energy_feature_names = [str(x) for x in e_feats_raw]
        temp_feature_names = [str(x) for x in t_feats_raw]

        # --- Validate against compiled constants ---
        _validate_feature_list(
            "energy",
            energy_feature_names,
            ENERGY_MODEL_FEATURES,
        )
        _validate_feature_list(
            "temperature",
            temp_feature_names,
            TEMP_MODEL_FEATURES,
        )

        # --- Extract hyperparams for metadata ---
        e_params = (
            energy_model.get_params()
            if hasattr(energy_model, "get_params")
            else {}
        )
        t_params = (
            temp_model.get_params()
            if hasattr(temp_model, "get_params")
            else {}
        )

        return cls(
            energy_model=energy_model,
            temp_model=temp_model,
            energy_feature_names=energy_feature_names,
            temp_feature_names=temp_feature_names,
            feature_list_pkl=fl,
            energy_n_estimators=e_params.get("n_estimators"),
            energy_learning_rate=e_params.get("learning_rate"),
            energy_max_depth=e_params.get("max_depth"),
            temp_n_estimators=t_params.get("n_estimators"),
            temp_learning_rate=t_params.get("learning_rate"),
            temp_max_depth=t_params.get("max_depth"),
        )


def _validate_feature_list(
    label: str,
    from_model: List[str],
    from_config: List[str],
) -> None:
    """Raise ``ModelLoadError`` if the model's feature names do not match
    the compiled constants in ``config.py``.

    This is an integrity check: it guards against the pkl files being
    swapped or regenerated with different feature engineering.
    """
    if len(from_model) != len(from_config):
        raise ModelLoadError(
            f"{label} model has {len(from_model)} features, "
            f"but config expects {len(from_config)}."
        )
    mismatches = [
        (i, m, c)
        for i, (m, c) in enumerate(zip(from_model, from_config))
        if m != c
    ]
    if mismatches:
        detail = "; ".join(
            f"[{i}] model={m!r} vs config={c!r}" for i, m, c in mismatches
        )
        raise ModelLoadError(
            f"{label} model feature ordering mismatch: {detail}"
        )
