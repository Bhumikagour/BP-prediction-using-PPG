"""
Copies the trained models and the held-out test set into this folder so the
deployment bundle is self-contained.

They are not duplicated in this folder by default because they already exist
next to your local backend, and keeping one copy avoids the two drifting apart.

Run once, from inside this folder:
    python prepare.py
"""
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(os.path.dirname(HERE), "pulseiq-backend", "ml_assets")
TARGET = os.path.join(HERE, "ml_assets")

# Everything the API loads at startup. Anything not listed here is training-time
# code the running service never imports.
NEEDED = {
    "data": ["processed_test_unseen.npz"],
    "models": [
        "best_model_SBP.joblib", "best_model_DBP.joblib", "best_model_MAP.joblib",
        "selected_features_SBP.joblib", "selected_features_DBP.joblib",
        "selected_features_MAP.joblib", "feature_scaler_StandardScaler.joblib",
        "best_resnet_bilstm_model.pt",
        # best_multimodal_resnet_bilstm.pt is deliberately NOT copied: the API
        # never loads it (the multimodal figures are reported from the project's
        # evaluation report, not recomputed), and at 11.5 MB it was the only
        # file that would have needed Git LFS.
    ],
    "src": [
        "config.py", "dl_model.py", "feature_engineering.py",
        "feature_selection.py", "metrics.py", "utils.py",
    ],
    "configs": ["config.py"],
}


def main():
    if not os.path.isdir(SOURCE):
        sys.exit(f"Could not find {SOURCE}\n"
                 f"Run this from the pulseiq-space folder, with pulseiq-backend "
                 f"as a sibling folder.")

    copied = missing = 0
    for sub, names in NEEDED.items():
        os.makedirs(os.path.join(TARGET, sub), exist_ok=True)
        for name in names:
            src = os.path.join(SOURCE, sub, name)
            if not os.path.exists(src):
                print(f"  MISSING  {sub}/{name}")
                missing += 1
                continue
            shutil.copy2(src, os.path.join(TARGET, sub, name))
            copied += 1

    # Remove anything a previous run left behind that is no longer needed --
    # notably the 11.5 MB multimodal checkpoint, which the API never loads and
    # which is the only file large enough to require Git LFS.
    keep = {os.path.join(TARGET, sub, n) for sub, names in NEEDED.items() for n in names}
    pruned = 0
    for root, _, files in os.walk(TARGET):
        for f in files:
            path = os.path.join(root, f)
            if path not in keep:
                os.remove(path)
                print(f"  removed   {os.path.relpath(path, TARGET)} (unused)")
                pruned += 1

    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(TARGET) for f in fs)
    print(f"\nCopied {copied} files, removed {pruned} unused, "
          f"{size / 1e6:.1f} MB total in ml_assets/")
    if missing:
        sys.exit(f"{missing} file(s) missing -- the container will fail to start "
                 f"without them.")
    print("Ready. Next: commit and push to GitHub, then deploy on Render (see README).")


if __name__ == "__main__":
    main()
