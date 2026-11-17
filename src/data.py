"""Dataset loading, label construction and feature engineering.

This module is the single source of truth shared by ``train.py`` (offline
training) and ``predict.py`` (online serving). Keeping the label rule and the
feature matrix in one place is what guarantees the served model receives
exactly the columns it was fitted on.
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

DATASET_URL = "https://archive.ics.uci.edu/static/public/601/data.csv"
DEFAULT_DATA_PATH = Path("data/raw/ai4i2020.csv")

#: Column in the dataset holding the manufacturer's ground-truth outcome:
#: 1 = the machine physically broke during production, 0 = it did not.
GROUND_TRUTH_COL = "Machine failure"

#: Name of the supervised target used by this project.
TARGET_COL = "quality_fail"

#: Human-readable definition of the target, embedded in the saved artifact.
TARGET_DEFINITION = (
    "quality_fail = 'Machine failure'  (1 = FAIL quality inspection, 0 = Pass). "
    "Ground truth 'Machine failure' is the manufacturer's own breakdown label; "
    "the quality verdict is derived from it as quality_pass = 1 - quality_fail."
)

#: Fault-type columns. These are recorded by the same inspection process that
#: sets ``Machine failure``, so they are post-hoc descriptions of the outcome
#: rather than upstream predictors of it. They are NEVER used as model inputs
#: and never used to build the label -- only for exploratory analysis.
FAULT_SUBTYPE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

#: Identifier columns with no predictive value.
ID_COLS = ["UID", "Product ID"]

#: Raw sensor columns read by the model.
SENSOR_COLS = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

# --------------------------------------------------------------------------
# Out-of-specification operating envelope
# --------------------------------------------------------------------------
# These bounds are taken from the *univariate* distribution of each feature
# (see the notebook, section 5.3). They express "was this process run inside
# its normal window?" and are computed WITHOUT reference to the label, so no
# target information leaks into the inputs.
OOS_BOUNDS = {
    "Air temperature": (296.0, 304.0),
    "Process temperature": (307.0, 313.0),
    "Torque": (12.0, 62.0),
    "Rotational speed": (1300.0, 2200.0),
}

#: Which feature each out-of-spec flag is derived from.
OOS_FLAG_FOR = {
    "Air temperature": "air_temp_oos",
    "Process temperature": "proc_temp_oos",
    "Torque": "torque_oos",
    "Rotational speed": "rot_speed_oos",
}

#: Final feature matrix columns, in order. Used as the contract between
#: training and serving.
FEATURE_NAMES = [
    "air_temp",
    "proc_temp",
    "rot_speed",
    "torque",
    "tool_wear",
    "temp_diff",
    "torque_per_krpm",
    "wear_per_load",
    "air_temp_oos",
    "proc_temp_oos",
    "torque_oos",
    "rot_speed_oos",
    "type_L",
    "type_M",
]


def download_dataset(dest: Path = DEFAULT_DATA_PATH) -> Path:
    """Download the UCI AI4I 2020 dataset to ``dest`` if it is not there yet."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset not found at {dest}. Downloading from {DATASET_URL} ...")
    with urlopen(DATASET_URL) as response:  # noqa: S310 - fixed, trusted URL
        payload = response.read()
    dest.write_bytes(payload)
    print(f"Saved dataset to {dest} ({len(payload):,} bytes)")
    return dest


def load_data(path: str | Path = DEFAULT_DATA_PATH, allow_download: bool = True) -> pd.DataFrame:
    """Read the raw AI4I CSV into a DataFrame.

    The UCI file is UTF-8 with a BOM, hence ``utf-8-sig``.
    """
    path = Path(path)
    if not path.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Download it from {DATASET_URL} "
                f"and place it at {path}."
            )
        path = download_dataset(path)
    return pd.read_csv(path, encoding="utf-8-sig")


# --------------------------------------------------------------------------
# Label
# --------------------------------------------------------------------------


def make_target(df: pd.DataFrame) -> pd.Series:
    """Build the supervised quality label from the dataset's ground truth.

    Ground truth : ``df["Machine failure"]`` -- the manufacturer's own label,
                   1 = the machine physically broke during production.
    Derived label: ``quality_fail`` mirrors that outcome from the quality
                   department's point of view:
                     1 -> "Fail"  (the product failed quality inspection)
                     0 -> "Pass"

    The fault-type columns (``TWF``/``HDF``/``PWF``/``OSF``/``RNF``) are never
    consulted: they are post-hoc classifications of the same outcome.
    """
    if GROUND_TRUTH_COL not in df.columns:
        raise KeyError(f"Expected {GROUND_TRUTH_COL!r} column in the dataset")
    return df[GROUND_TRUTH_COL].astype(int).rename(TARGET_COL)


def make_pass_label(df: pd.DataFrame) -> pd.Series:
    """The complementary view: 1 = the product passed quality inspection."""
    return (1 - make_target(df)).rename("quality_pass")


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the model input matrix.

    The result always has exactly ``FEATURE_NAMES`` columns in that order,
    and works on a single-row DataFrame (which is how the API uses it).

    The label is built separately by :func:`make_target` so that no target
    information can ever enter this function.
    """
    out = pd.DataFrame(index=df.index)

    # --- raw sensors ---
    out["air_temp"] = df["Air temperature"].astype(float)
    out["proc_temp"] = df["Process temperature"].astype(float)
    out["rot_speed"] = df["Rotational speed"].astype(float)
    out["torque"] = df["Torque"].astype(float)
    out["tool_wear"] = df["Tool wear"].astype(float)

    # --- derived process relationships ---
    # Heat actually delivered to the workpiece, not just ambient temperature.
    out["temp_diff"] = out["proc_temp"] - out["air_temp"]
    # Mechanical load per unit of speed: flags a workpiece too heavy for the rpm.
    out["torque_per_krpm"] = out["torque"] / (out["rot_speed"] / 1000.0)
    # Tool wear accumulated per unit of work done.
    out["wear_per_load"] = out["tool_wear"] / (out["torque"] + 1.0)

    # --- out-of-spec tolerance flags (inputs only, never part of the label) ---
    for col, (low, high) in OOS_BOUNDS.items():
        flag = OOS_FLAG_FOR[col]
        out[flag] = ((df[col].astype(float) < low) | (df[col].astype(float) > high)).astype(int)

    # --- product type (H is the reference level) ---
    types = df["Type"].astype(str).str.upper()
    out["type_L"] = (types == "L").astype(int)
    out["type_M"] = (types == "M").astype(int)

    return out[FEATURE_NAMES]
