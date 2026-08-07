"""
Central Configuration Module for MIMIC-IV Classical ML Blood Pressure Estimation.
Defines paths, feature extraction parameters, GroupKFold cross-validation settings,
hyperparameter grids, and model options.
"""

import os
from typing import Dict, Any, List

class Config:
    # Base Project Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR: str = BASE_DIR
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    REPORTS_DIR: str = os.path.join(BASE_DIR, "reports")
    FIGURES_DIR: str = os.path.join(REPORTS_DIR, "figures")
    CONFIGS_DIR: str = os.path.join(BASE_DIR, "configs")
    
    # Dataset File Paths
    TRAIN_NPZ_PATH: str = os.path.join(DATA_DIR, "processed_train_mimic4.npz")
    TEST_NPZ_PATH: str = os.path.join(DATA_DIR, "processed_test_unseen.npz")

    # Global Random Seed
    RANDOM_SEED: int = 42

    # Signal Characteristics
    SAMPLING_RATE: float = 125.0  # Hz
    WINDOW_SEC: float = 12.0
    WINDOW_SAMPLES: int = int(SAMPLING_RATE * WINDOW_SEC)  # 1500 samples

    # Target Names
    TARGET_NAMES: List[str] = ["SBP", "DBP", "MAP"]

    # Feature Processing Parameters
    CORRELATION_THRESHOLD: float = 0.95  # Max collinearity threshold
    VARIANCE_THRESHOLD: float = 1e-4      # Near-zero variance threshold
    TOP_K_FEATURES: int = 25              # Number of features selected per target

    # Cross Validation Settings
    N_SPLITS_CV: int = 5  # GroupKFold splits (grouped by subject_id)

    # Models to Evaluate
    MODELS_TO_RUN: List[str] = [
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "HuberRegressor",
        "BayesianRidge",
        "DecisionTree",
        "RandomForest",
        "ExtraTrees",
        "GradientBoosting",
        "AdaBoost",
        "SVR",
        "KNN",
        "XGBoost",
        "LightGBM"
    ]

    # Hyperparameter Search Grids
    PARAM_GRIDS: Dict[str, Dict[str, Any]] = {
        "Ridge": {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
        },
        "Lasso": {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        "ElasticNet": {
            "alpha": [0.001, 0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.7, 0.9]
        },
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10]
        },
        "ExtraTrees": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        },
        "GradientBoosting": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        },
        "SVR": {
            "C": [0.1, 1.0, 10.0, 100.0],
            "epsilon": [0.01, 0.1, 0.2],
            "kernel": ["rbf", "linear"]
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 11, 15],
            "weights": ["uniform", "distance"]
        },
        "XGBoost": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7]
        },
        "LightGBM": {
            "n_estimators": [50, 100, 150],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63]
        }
    }

    @classmethod
    def ensure_directories(cls):
        """Creates required directories if they do not exist."""
        for path in [cls.MODELS_DIR, cls.REPORTS_DIR, cls.FIGURES_DIR, cls.CONFIGS_DIR]:
            os.makedirs(path, exist_ok=True)
