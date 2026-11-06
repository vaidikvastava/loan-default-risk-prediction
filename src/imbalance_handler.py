"""
Imbalanced learning module for credit risk modeling using Imbalanced-Learn.
Provides SMOTE oversampling, Random Under-Sampling, and hybrid resampling strategies.
"""
import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    logger.warning("imbalanced-learn not installed. Run 'pip install imbalanced-learn' to enable SMOTE/ADASYN.")


class ImbalanceHandler:
    """
    Manages resampling techniques to treat severe class imbalance (~12:1 negative-to-positive ratio).
    """

    def __init__(self, method: str = "smote", sampling_strategy: float = 0.5, random_state: int = 42):
        self.method = method.lower()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Resamples feature matrix and target series using the configured imbalance technique.
        """
        if not HAS_IMBLEARN:
            logger.info("Imbalanced-learn unavailable; preserving original data and recommending algorithmic weighting.")
            return X, y

        orig_dist = y.value_counts().to_dict()
        logger.info(f"Original distribution: {orig_dist}")

        if self.method == "smote":
            sampler = SMOTE(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        elif self.method == "adasyn":
            sampler = ADASYN(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        elif self.method == "undersample":
            sampler = RandomUnderSampler(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        elif self.method == "hybrid":
            sampler = SMOTETomek(sampling_strategy=self.sampling_strategy, random_state=self.random_state)
        else:
            logger.warning(f"Unknown resampling method '{self.method}'; returning original data.")
            return X, y

        X_res, y_res = sampler.fit_resample(X, y)
        res_dist = pd.Series(y_res).value_counts().to_dict()
        logger.info(f"Resampled distribution via {self.method.upper()}: {res_dist}")

        # Return as DataFrames with column names preserved
        if isinstance(X, pd.DataFrame):
            X_res = pd.DataFrame(X_res, columns=X.columns)
        if isinstance(y, pd.Series):
            y_res = pd.Series(y_res, name=y.name)

        return X_res, y_res
