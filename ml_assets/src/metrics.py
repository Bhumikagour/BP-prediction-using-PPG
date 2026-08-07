"""
Metrics Module for Classical ML Blood Pressure Estimation.
Computes classical regression metrics (MAE, RMSE, R2, MAPE, Pearson r),
alongside clinical AAMI SP10 standards (Mean Error <= 5 mmHg, SDE <= 8 mmHg)
and British Hypertension Society (BHS) grading (Classes A, B, C, D).
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics, error bias, and correlations.
    """
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()

    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    me = float(np.mean(errors))          # Mean Error / Bias
    sde = float(np.std(errors))          # Standard Deviation of Error
    med_ae = float(np.median(abs_errors))
    mape = float(np.mean(abs_errors / (y_true + 1e-8)) * 100.0)

    # R2 Score
    ss_tot = np.sum(np.square(y_true - np.mean(y_true)))
    ss_res = np.sum(np.square(errors))
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    # Correlations
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_r, _ = stats.spearmanr(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "MedAE": med_ae,
        "ME": me,
        "SDE": sde,
        "Pearson_R": float(pearson_r),
        "Spearman_R": float(spearman_r)
    }

def evaluate_aami_sp10(y_true: np.ndarray, y_pred: np.ndarray, target_name: str = "Target") -> Dict[str, Any]:
    """
    Evaluates compliance against AAMI SP10 Standard:
    - Mean Error (ME) <= 5.0 mmHg
    - Standard Deviation of Error (SDE) <= 8.0 mmHg
    """
    metrics = compute_regression_metrics(y_true, y_pred)
    me = abs(metrics["ME"])
    sde = metrics["SDE"]

    pass_me = me <= 5.0
    pass_sde = sde <= 8.0
    pass_aami = pass_me and pass_sde

    return {
        "target": target_name,
        "mean_error": metrics["ME"],
        "std_error": sde,
        "pass_me": pass_me,
        "pass_sde": pass_sde,
        "aami_compliant": pass_aami,
        "status": "PASS (AAMI SP10 Compliant)" if pass_aami else "FAIL (Exceeds AAMI Limits)"
    }

def evaluate_bhs_grade(y_true: np.ndarray, y_pred: np.ndarray, target_name: str = "Target") -> Dict[str, Any]:
    """
    Evaluates British Hypertension Society (BHS) Performance Grade:
    Calculates cumulative percentage of absolute errors <= 5 mmHg, <= 10 mmHg, <= 15 mmHg.
    Grade A: >=60% (<=5), >=85% (<=10), >=95% (<=15)
    Grade B: >=50% (<=5), >=75% (<=10), >=90% (<=15)
    Grade C: >=40% (<=5), >=65% (<=10), >=85% (<=15)
    Grade D: Worse than C
    """
    abs_errors = np.abs(y_pred - y_true)
    total_n = len(abs_errors)

    pct_5 = float(np.sum(abs_errors <= 5.0) / total_n * 100.0)
    pct_10 = float(np.sum(abs_errors <= 10.0) / total_n * 100.0)
    pct_15 = float(np.sum(abs_errors <= 15.0) / total_n * 100.0)

    # Determine BHS Grade
    if pct_5 >= 60.0 and pct_10 >= 85.0 and pct_15 >= 95.0:
        grade = "Grade A"
    elif pct_5 >= 50.0 and pct_10 >= 75.0 and pct_15 >= 90.0:
        grade = "Grade B"
    elif pct_5 >= 40.0 and pct_10 >= 65.0 and pct_15 >= 85.0:
        grade = "Grade C"
    else:
        grade = "Grade D"

    return {
        "target": target_name,
        "pct_le_5mmHg": pct_5,
        "pct_le_10mmHg": pct_10,
        "pct_le_15mmHg": pct_15,
        "bhs_grade": grade
    }
