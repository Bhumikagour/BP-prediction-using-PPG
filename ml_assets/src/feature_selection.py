"""
Feature Cleaning & Selection Module for Classical ML Blood Pressure Estimation.
Performs automated feature cleaning (NaN imputation, constant/near-zero variance removal,
high correlation filtering >0.95), feature scaling, and feature selection (Mutual Info & Random Forest).
"""

import os
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from src.config import Config
from src.utils import setup_logger, save_object

logger = setup_logger("FeatureSelection")

def clean_feature_matrix(df: pd.DataFrame, corr_threshold: float = Config.CORRELATION_THRESHOLD) -> pd.DataFrame:
    """
    Cleans raw handcrafted features:
    1. Imputes NaNs / Infs with column medians.
    2. Drops constant and near-zero variance features.
    3. Removes highly correlated collinear features (correlation > corr_threshold).
    """
    logger.info(f"Cleaning feature matrix (Original shape: {df.shape})...")

    # 1. Fill NaNs / Infs with column median
    df_clean = df.replace([np.inf, -np.inf], np.nan).copy()
    df_clean = df_clean.fillna(df_clean.median(numeric_only=True)).fillna(0.0)

    # 2. Drop constant features
    constant_cols = [col for col in df_clean.columns if df_clean[col].std() <= 1e-6]
    if constant_cols:
        logger.info(f"  Dropping {len(constant_cols)} constant features...")
        df_clean = df_clean.drop(columns=constant_cols)

    # 3. High Correlation Filter (> corr_threshold)
    logger.info(f"  Filtering highly correlated features (> {corr_threshold})...")
    corr_matrix = df_clean.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > corr_threshold)]

    if to_drop:
        logger.info(f"  Dropping {len(to_drop)} collinear features...")
        df_clean = df_clean.drop(columns=to_drop)

    logger.info(f"Feature Cleaning Complete! Cleaned shape: {df_clean.shape}")
    return df_clean

def fit_scaler(X_train: pd.DataFrame, scaler_type: str = "StandardScaler") -> Tuple[Any, pd.DataFrame]:
    """
    Fits scaler ONLY on training data to prevent data leakage.
    Returns fitted scaler and scaled training DataFrame.
    """
    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_scaled_np = scaler.fit_transform(X_train)
    X_scaled_df = pd.DataFrame(X_scaled_np, columns=X_train.columns, index=X_train.index)

    return scaler, X_scaled_df

def select_top_k_features(
    X_train_df: pd.DataFrame,
    y_target: np.ndarray,
    target_name: str,
    top_k: int = Config.TOP_K_FEATURES,
    seed: int = Config.RANDOM_SEED
) -> Tuple[List[str], pd.DataFrame]:
    """
    Ranks features using Mutual Information and Random Forest Importance to pick the optimal top K features.
    """
    logger.info(f"Selecting top {top_k} features for target [{target_name}]...")

    # Mutual Information Scores
    mi_scores = mutual_info_regression(X_train_df, y_target, random_state=seed)
    mi_series = pd.Series(mi_scores, index=X_train_df.columns)

    # Random Forest Importance Scores
    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_train_df, y_target)
    rf_series = pd.Series(rf.feature_importances_, index=X_train_df.columns)

    # Composite Ranking (Rank Average)
    mi_rank = mi_series.rank(ascending=False)
    rf_rank = rf_series.rank(ascending=False)
    composite_rank = (mi_rank + rf_rank) / 2.0

    ranking_df = pd.DataFrame({
        "feature": X_train_df.columns,
        "mi_score": mi_series.values,
        "rf_importance": rf_series.values,
        "composite_rank": composite_rank.values
    }).sort_values("composite_rank")

    selected_features = ranking_df.head(top_k)["feature"].tolist()
    logger.info(f"Selected {len(selected_features)} features for [{target_name}]. Top 5: {selected_features[:5]}")

    return selected_features, ranking_df

def process_feature_selection(
    df_train_raw: pd.DataFrame,
    df_test_raw: pd.DataFrame,
    Y_train: np.ndarray,
    Y_test: np.ndarray,
    scaler_type: str = "StandardScaler"
) -> Dict[str, Any]:
    """
    End-to-End Feature Selection & Scaling Pipeline:
    1. Cleans train and test feature matrices.
    2. Fits scaler strictly on train.
    3. Selects top K features independently for SBP, DBP, MAP.
    4. Saves scalers and feature lists to models/ directory.
    """
    Config.ensure_directories()
    
    # 1. Clean feature matrix
    df_train_clean = clean_feature_matrix(df_train_raw)
    
    # Align test set columns to train clean columns
    common_cols = df_train_clean.columns.tolist()
    df_test_clean = df_test_raw.reindex(columns=common_cols).fillna(0.0)

    # 2. Fit Scaler strictly on train
    scaler, df_train_scaled = fit_scaler(df_train_clean, scaler_type=scaler_type)
    
    # Transform test set using saved scaler
    df_test_scaled_np = scaler.transform(df_test_clean)
    df_test_scaled = pd.DataFrame(df_test_scaled_np, columns=common_cols, index=df_test_clean.index)

    # Save fitted scaler
    scaler_path = os.path.join(Config.MODELS_DIR, f"feature_scaler_{scaler_type}.joblib")
    save_object(scaler, scaler_path)
    logger.info(f"Saved feature scaler to: {scaler_path}")

    selected_features_dict = {}
    rankings_dict = {}
    train_selected_dict = {}
    test_selected_dict = {}

    for target_idx, target_name in enumerate(Config.TARGET_NAMES):
        y_train_target = Y_train[:, target_idx]
        
        # Select top K features
        top_features, ranking_df = select_top_k_features(
            df_train_scaled, y_train_target, target_name, top_k=Config.TOP_K_FEATURES
        )

        selected_features_dict[target_name] = top_features
        rankings_dict[target_name] = ranking_df

        train_selected_dict[target_name] = df_train_scaled[top_features]
        test_selected_dict[target_name] = df_test_scaled[top_features]

        # Save selected feature list
        feats_path = os.path.join(Config.MODELS_DIR, f"selected_features_{target_name}.joblib")
        save_object(top_features, feats_path)

    return {
        "scaler": scaler,
        "clean_columns": common_cols,
        "selected_features": selected_features_dict,
        "feature_rankings": rankings_dict,
        "X_train_scaled": df_train_scaled,
        "X_test_scaled": df_test_scaled,
        "train_selected": train_selected_dict,
        "test_selected": test_selected_dict
    }
