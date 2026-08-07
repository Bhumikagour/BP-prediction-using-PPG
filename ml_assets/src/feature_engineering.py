"""
Feature Engineering Module for Classical ML Blood Pressure Estimation.
Extracts comprehensive physiological, statistical, morphological, derivative (APG/VPG),
and frequency-domain features from PPG, VPG, and APG waveform windows.
"""

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import pandas as pd
from typing import Dict, List, Tuple
from src.config import Config
from src.utils import setup_logger

logger = setup_logger("FeatureEngineering")

def extract_time_domain_features(sig: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    """Extracts Time Domain & Heart Rate Variability features."""
    feats = {}
    min_dist = int(0.4 * fs)
    peaks, _ = signal.find_peaks(sig, distance=min_dist, prominence=0.15 * np.std(sig))

    if len(peaks) >= 2:
        intervals = np.diff(peaks) / fs  # seconds
        hr_bpm = 60.0 / (np.mean(intervals) + 1e-8)
        feats[f"{prefix}_hr_bpm"] = float(hr_bpm)
        feats[f"{prefix}_pulse_interval_mean"] = float(np.mean(intervals))
        feats[f"{prefix}_pulse_interval_std"] = float(np.std(intervals))
        
        # HRV proxies
        if len(intervals) >= 2:
            rmssd = np.sqrt(np.mean(np.square(np.diff(intervals))))
            feats[f"{prefix}_hrv_rmssd"] = float(rmssd)
            feats[f"{prefix}_hrv_sdnn"] = float(np.std(intervals))
        else:
            feats[f"{prefix}_hrv_rmssd"] = 0.0
            feats[f"{prefix}_hrv_sdnn"] = 0.0
    else:
        feats[f"{prefix}_hr_bpm"] = 75.0
        feats[f"{prefix}_pulse_interval_mean"] = 0.8
        feats[f"{prefix}_pulse_interval_std"] = 0.0
        feats[f"{prefix}_hrv_rmssd"] = 0.0
        feats[f"{prefix}_hrv_sdnn"] = 0.0

    # Pulse Widths at 25%, 50%, 75%, 90%
    amp_max = np.max(sig)
    amp_min = np.min(sig)
    amp_range = amp_max - amp_min + 1e-8

    for pct in [25, 50, 75, 90]:
        thresh = amp_min + (pct / 100.0) * amp_range
        above_thresh = np.sum(sig >= thresh) / fs
        feats[f"{prefix}_pulse_width_{pct}"] = float(above_thresh)

    # Rise & Decay Time
    peak_idx = np.argmax(sig)
    valley_idx = np.argmin(sig)
    feats[f"{prefix}_rise_time"] = float(abs(peak_idx - valley_idx) / fs)
    feats[f"{prefix}_decay_time"] = float((len(sig) - peak_idx) / fs)
    feats[f"{prefix}_duty_cycle"] = float(feats[f"{prefix}_rise_time"] / (len(sig) / fs + 1e-8))

    return feats

def extract_amplitude_features(sig: np.ndarray, prefix: str) -> Dict[str, float]:
    """Extracts Amplitude & Energy features."""
    feats = {}
    p_max = np.max(sig)
    p_min = np.min(sig)
    p_mean = np.mean(sig)

    feats[f"{prefix}_amplitude_peak"] = float(p_max)
    feats[f"{prefix}_amplitude_valley"] = float(p_min)
    feats[f"{prefix}_pulse_amplitude"] = float(p_max - p_min)
    feats[f"{prefix}_peak_to_valley_diff"] = float(p_max - p_min)
    feats[f"{prefix}_mean_amplitude"] = float(p_mean)
    feats[f"{prefix}_median_amplitude"] = float(np.median(sig))

    # Area & Energy
    feats[f"{prefix}_area_under_pulse"] = float(np.trapezoid(sig))
    feats[f"{prefix}_area_above_baseline"] = float(np.trapezoid(sig - p_min))
    feats[f"{prefix}_signal_energy"] = float(np.sum(np.square(sig)))
    feats[f"{prefix}_signal_power"] = float(np.mean(np.square(sig)))

    return feats

def extract_morphological_features(ppg: np.ndarray, fs: float) -> Dict[str, float]:
    """Extracts Morphological Pulse Waveform features (Stiffness Index, Reflection Index, Augmentation Index)."""
    feats = {}
    p_max_idx = np.argmax(ppg)
    p_max = ppg[p_max_idx]

    # Systolic Peak Location
    feats["ppg_systolic_peak_location"] = float(p_max_idx / fs)

    # Dicrotic Notch Location (find local minimum after systolic peak)
    if p_max_idx < len(ppg) - 10:
        post_peak = ppg[p_max_idx:]
        notch_relative = np.argmin(post_peak)
        notch_idx = p_max_idx + notch_relative
        dicrotic_val = ppg[notch_idx]
        feats["ppg_dicrotic_notch_location"] = float(notch_idx / fs)
        feats["ppg_reflection_index"] = float((dicrotic_val / (p_max + 1e-8)) * 100.0)
    else:
        feats["ppg_dicrotic_notch_location"] = float(p_max_idx / fs)
        feats["ppg_reflection_index"] = 50.0

    # Stiffness Index (Height / delta T)
    dt = feats["ppg_dicrotic_notch_location"] - feats["ppg_systolic_peak_location"]
    feats["ppg_stiffness_index"] = float(1.75 / (dt + 1e-3))  # assumed height = 1.75m
    feats["ppg_augmentation_index"] = float((p_max - ppg[0]) / (p_max + 1e-8))

    # Crest Factor & Shape Factor
    rms = np.sqrt(np.mean(np.square(ppg)))
    feats["ppg_crest_factor"] = float(p_max / (rms + 1e-8))
    feats["ppg_pulse_shape_factor"] = float(np.mean(ppg) / (p_max + 1e-8))

    return feats

def extract_apg_vpg_features(vpg: np.ndarray, apg: np.ndarray, fs: float) -> Dict[str, float]:
    """Extracts Derivative Features from VPG (1st deriv) and APG (2nd deriv, waves a, b, c, d, e)."""
    feats = {}

    # VPG Features
    feats["vpg_max_derivative"] = float(np.max(vpg))
    feats["vpg_min_derivative"] = float(np.min(vpg))
    feats["vpg_mean_derivative"] = float(np.mean(vpg))
    feats["vpg_energy"] = float(np.sum(np.square(vpg)))
    zero_cross = np.where(np.diff(np.signbit(vpg)))[0]
    feats["vpg_zero_crossings"] = float(len(zero_cross))

    # APG Wave Features (a, b, c, d, e)
    # Wave a: initial positive peak
    # Wave b: early negative trough
    # Wave c: re-elevated peak
    # Wave d: late negative trough
    # Wave e: diastolic peak
    a_val = np.max(apg)
    b_val = np.min(apg[:len(apg)//2]) if len(apg) > 10 else -0.5
    c_val = np.max(apg[len(apg)//4:len(apg)//2]) if len(apg) > 20 else 0.2
    d_val = np.min(apg[len(apg)//2:3*len(apg)//4]) if len(apg) > 20 else -0.2
    e_val = np.max(apg[3*len(apg)//4:]) if len(apg) > 20 else 0.1

    feats["apg_wave_a"] = float(a_val)
    feats["apg_wave_b"] = float(b_val)
    feats["apg_wave_c"] = float(c_val)
    feats["apg_wave_d"] = float(d_val)
    feats["apg_wave_e"] = float(e_val)

    # Ratios
    a_denom = a_val if abs(a_val) > 1e-6 else 1.0
    feats["apg_b_a_ratio"] = float(b_val / a_denom)
    feats["apg_c_a_ratio"] = float(c_val / a_denom)
    feats["apg_d_a_ratio"] = float(d_val / a_denom)
    feats["apg_e_a_ratio"] = float(e_val / a_denom)

    # Aging Index = (b - c - d - e) / a
    feats["apg_aging_index"] = float((b_val - c_val - d_val - e_val) / a_denom)
    feats["apg_energy"] = float(np.sum(np.square(apg)))

    return feats

def extract_statistical_features(sig: np.ndarray, prefix: str) -> Dict[str, float]:
    """Extracts Statistical Moments, Skewness, Kurtosis, IQR, and MAD."""
    feats = {}
    std_val = np.std(sig)
    mean_val = np.mean(sig)

    feats[f"{prefix}_stat_mean"] = float(mean_val)
    feats[f"{prefix}_stat_std"] = float(std_val)
    feats[f"{prefix}_stat_var"] = float(np.var(sig))
    feats[f"{prefix}_stat_rms"] = float(np.sqrt(np.mean(np.square(sig))))
    feats[f"{prefix}_stat_skew"] = float(stats.skew(sig))
    feats[f"{prefix}_stat_kurtosis"] = float(stats.kurtosis(sig))

    q25, q50, q75 = np.percentile(sig, [25, 50, 75])
    feats[f"{prefix}_stat_iqr"] = float(q75 - q25)
    feats[f"{prefix}_stat_mad"] = float(np.median(np.abs(sig - q50)))
    feats[f"{prefix}_stat_cov"] = float(std_val / (abs(mean_val) + 1e-8))

    for p in [10, 25, 75, 90]:
        feats[f"{prefix}_stat_p{p}"] = float(np.percentile(sig, p))

    return feats

def extract_frequency_features(sig: np.ndarray, fs: float, prefix: str) -> Dict[str, float]:
    """Extracts Spectral FFT, Spectral Centroid, Band Energies, and LF/HF Ratio."""
    feats = {}
    freqs, psd = signal.welch(sig, fs=fs, nperseg=min(len(sig), 256))

    total_power = np.sum(psd) + 1e-8
    feats[f"{prefix}_total_spectral_power"] = float(total_power)

    # Dominant Frequency
    dom_idx = np.argmax(psd)
    feats[f"{prefix}_dominant_frequency"] = float(freqs[dom_idx])

    # Spectral Centroid & Roll-off
    feats[f"{prefix}_spectral_centroid"] = float(np.sum(freqs * psd) / total_power)
    cum_power = np.cumsum(psd) / total_power
    rolloff_idx = np.where(cum_power >= 0.85)[0]
    feats[f"{prefix}_spectral_rolloff"] = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else float(freqs[-1])

    # Spectral Entropy
    psd_norm = psd / total_power
    psd_norm = psd_norm[psd_norm > 0]
    feats[f"{prefix}_spectral_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm)))

    # Band Energies (Low Frequency 0.5-4 Hz vs High Frequency 4-12 Hz)
    lf_mask = (freqs >= 0.5) & (freqs <= 4.0)
    hf_mask = (freqs > 4.0) & (freqs <= 12.0)
    lf_energy = np.sum(psd[lf_mask])
    hf_energy = np.sum(psd[hf_mask])

    feats[f"{prefix}_band_energy_lf"] = float(lf_energy)
    feats[f"{prefix}_band_energy_hf"] = float(hf_energy)
    feats[f"{prefix}_lf_hf_ratio"] = float(lf_energy / (hf_energy + 1e-8))

    return feats

def extract_hjorth_features(sig: np.ndarray, prefix: str) -> Dict[str, float]:
    """Extracts Hjorth Parameters: Activity, Mobility, and Complexity."""
    feats = {}
    d1 = np.diff(sig)
    d2 = np.diff(d1)

    var_s = np.var(sig)
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)

    activity = var_s
    mobility = np.sqrt(var_d1 / (var_s + 1e-8))
    complexity = np.sqrt(var_d2 / (var_d1 + 1e-8)) / (mobility + 1e-8)

    feats[f"{prefix}_hjorth_activity"] = float(activity)
    feats[f"{prefix}_hjorth_mobility"] = float(mobility)
    feats[f"{prefix}_hjorth_complexity"] = float(complexity)

    return feats

def extract_window_features(ppg: np.ndarray, vpg: np.ndarray, apg: np.ndarray, fs: float) -> Dict[str, float]:
    """Combines all handcrafted features for a single 12-second window."""
    window_feats = {}

    # 1. PPG Channel Features
    window_feats.update(extract_time_domain_features(ppg, fs, "ppg"))
    window_feats.update(extract_amplitude_features(ppg, "ppg"))
    window_feats.update(extract_morphological_features(ppg, fs))
    window_feats.update(extract_statistical_features(ppg, "ppg"))
    window_feats.update(extract_frequency_features(ppg, fs, "ppg"))
    window_feats.update(extract_hjorth_features(ppg, "ppg"))

    # 2. VPG Channel Features
    window_feats.update(extract_amplitude_features(vpg, "vpg"))
    window_feats.update(extract_statistical_features(vpg, "vpg"))
    window_feats.update(extract_frequency_features(vpg, fs, "vpg"))
    window_feats.update(extract_hjorth_features(vpg, "vpg"))

    # 3. APG Channel Features
    window_feats.update(extract_apg_vpg_features(vpg, apg, fs))
    window_feats.update(extract_statistical_features(apg, "apg"))
    window_feats.update(extract_frequency_features(apg, fs, "apg"))
    window_feats.update(extract_hjorth_features(apg, "apg"))

    return window_feats

def extract_dataset_features(X_matrix: np.ndarray, fs: float = Config.SAMPLING_RATE) -> pd.DataFrame:
    """
    Extracts features for an entire dataset tensor matrix of shape [N, 3, 1500].
    Channels: 0=PPG, 1=VPG, 2=APG.
    """
    n_samples = len(X_matrix)
    logger.info(f"Extracting handcrafted features for {n_samples} windows...")

    feature_list = []
    for i in range(n_samples):
        ppg = X_matrix[i, 0, :]
        vpg = X_matrix[i, 1, :]
        apg = X_matrix[i, 2, :]

        feats = extract_window_features(ppg, vpg, apg, fs)
        feature_list.append(feats)

        if (i + 1) % 250 == 0 or (i + 1) == n_samples:
            logger.info(f"  Extracted features for {i + 1}/{n_samples} windows...")

    df_features = pd.DataFrame(feature_list)
    logger.info(f"Feature Extraction Complete! Extracted {df_features.shape[1]} features per window.")

    return df_features
