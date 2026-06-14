"""Offline accuracy experiments for c-VEP decoder settings.

This module is intentionally independent from the Dareplane control room and
online LSL loop. It loads recorded XDF training data, evaluates decoder settings
with cross-validation, and writes a CSV so parameter changes can be compared.
"""

from __future__ import annotations

import copy
import csv
from itertools import product
from pathlib import Path

import numpy as np
import toml
from dareplane_utils.signal_processing.filtering import FilterBank
from fire import Fire
from scipy.signal import resample

from cvep_decoder.train_decoder import (
    calc_cv_accuracy_early_stop,
    classifier_meta_from_cfg,
    get_training_data_files,
    load_raw_and_events,
)
from cvep_decoder.utils.logging import logger


def _parse_floats(value: str | float | int | tuple | list | None, default: list[float]) -> list[float]:
    if value is None:
        return default
    if isinstance(value, int | float):
        return [float(value)]
    if isinstance(value, tuple | list):
        return [float(v) for v in value]
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _parse_strings(value: str | tuple | list | None, default: list[str]) -> list[str]:
    if value is None:
        return default
    if isinstance(value, tuple | list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_passbands(value: str | tuple | list | None, default: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if value is None:
        return default
    if isinstance(value, tuple | list):
        bands = []
        for item in value:
            if isinstance(item, tuple | list):
                if len(item) != 2:
                    raise ValueError(f"Invalid passband {item}. Expected two values.")
                bands.append((float(item[0]), float(item[1])))
            else:
                low_high = str(item).replace(":", "-").split("-")
                if len(low_high) != 2:
                    raise ValueError(
                        f"Invalid passband '{item}'. Use formats like '6-40,2-30'."
                    )
                bands.append((float(low_high[0]), float(low_high[1])))
        return bands

    bands = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        low_high = item.replace(":", "-").split("-")
        if len(low_high) != 2:
            raise ValueError(
                f"Invalid passband '{item}'. Use formats like '6-40,2-30'."
            )
        bands.append((float(low_high[0]), float(low_high[1])))
    return bands


def _load_epochs_and_labels(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    cmeta = classifier_meta_from_cfg(cfg)
    t_files = get_training_data_files(cfg)
    if len(t_files) == 0:
        raise FileNotFoundError("No training files found for offline accuracy experiment.")

    eeg_list = []
    lbl_list = []

    for t_file in t_files:
        x, events, sfreq = load_raw_and_events(
            fpath=t_file,
            marker_stream_name=cfg["streams"]["marker_stream_name"],
            data_stream_name=cfg["streams"]["data_stream_name"],
            selected_channels=cfg["data"].get("selected_channels", None),
        )

        fb = FilterBank(
            bands={"band": cmeta.fband},
            sfreq=sfreq,
            output="signal",
            n_in_channels=x.shape[1],
            filter_buffer_s=np.ceil(x.shape[0] / sfreq),
        )
        fb.filter(x, np.arange(x.shape[0]) / sfreq)
        xf = fb.get_data()[:, :, 0]

        onsets = events[events[:, 1] == cfg["stimulus"]["trial_marker"], 0]
        if cfg["streams"]["padding_size_s"] is not None and cfg["streams"]["padding_size_s"] > 0:
            onsets -= int(cfg["streams"]["padding_size_s"] * sfreq)

        eeg_list += [
            xf[t - int(cmeta.tmin * sfreq):t + int(cmeta.tmax * sfreq), :]
            for t in onsets
        ]

        lbl_list += [
            int(m.split(";")[1].split("=")[1])
            for m in events[:, 1]
            if m.startswith(cfg["stimulus"]["cue_marker"])
        ]

    if len(eeg_list) != len(lbl_list):
        raise ValueError(
            f"Found {len(eeg_list)} EEG trials but {len(lbl_list)} labels. "
            "Check trial/cue markers in the XDF files."
        )

    X = np.stack(eeg_list, axis=0).transpose(0, 2, 1)
    y = np.stack(lbl_list, axis=0)

    X = resample(X, num=int((cmeta.tmax - cmeta.tmin) * cmeta.sfreq), axis=2)

    if cfg["streams"]["padding_size_s"] is not None and cfg["streams"]["padding_size_s"] > 0:
        pad = int(cfg["streams"]["padding_size_s"] * cmeta.sfreq)
        X = X[:, :, pad:]

    return X, y


def _load_stimulus(cfg: dict, use_full_codebook: bool) -> np.ndarray:
    cmeta = classifier_meta_from_cfg(cfg)
    V = np.repeat(
        np.load(cfg["training"]["codes_file"])["codes"],
        int(cmeta.sfreq / cmeta.presentation_rate),
        axis=1,
    )

    n_keys = int(cfg["stimulus"]["n_keys"])
    if not use_full_codebook and n_keys > 0:
        V = V[:n_keys, :]

    return V


def _validate_labels_match_stimulus(y: np.ndarray, V: np.ndarray, use_full_codebook: bool) -> None:
    max_label = int(np.max(y))
    if max_label < V.shape[0]:
        return

    hint = (
        "Your labels contain class IDs from the full keyboard, but this run only "
        f"loaded {V.shape[0]} stimulus codes. "
    )
    if not use_full_codebook:
        hint += (
            "If this is old full-keyboard data, rerun with "
            "`--use_full_codebook=True`. If this is for the car task, record new "
            "W/A/S/D-only training data after setting `n_keys = 4` and the "
            "4-key layout in `speller.toml`."
        )
    else:
        hint += (
            "Even the full codebook is smaller than the labels in the data. "
            "Check that `training.codes_file` matches the stimulus codes used "
            "when the XDF was recorded."
        )
    raise ValueError(f"Maximum label is {max_label}, stimulus codes={V.shape[0]}. {hint}")


def run_grid(
    config_path: Path = Path("./configs/decoder_unity.toml"),
    output_csv: Path = Path("./offline_accuracy_results.csv"),
    passband_grid: str | None = None,
    tmin_grid: str | None = None,
    encoding_grid: str | None = None,
    stopping_grid: str | None = None,
    min_time_grid: str | None = None,
    target_accuracy_grid: str | None = None,
    n_folds: int = 4,
    use_full_codebook: bool = False,
) -> list[dict]:
    """Run a cross-validated parameter grid on recorded training data.

    Grid arguments are comma-separated strings. Examples:
    passband_grid="6-40,2-30", tmin_grid="0.0,0.05,0.1".
    """

    config_path = Path(config_path)
    output_csv = Path(output_csv)

    base_cfg = toml.load(config_path)

    passbands = _parse_passbands(
        passband_grid,
        [tuple(base_cfg["data"]["passband_hz"])],
    )
    tmins = _parse_floats(tmin_grid, [float(base_cfg["decoder"]["tmin_s"])])
    encodings = _parse_floats(
        encoding_grid,
        [float(base_cfg["decoder"]["encoding_length_s"])],
    )
    stoppings = _parse_strings(stopping_grid, [base_cfg["decoder"]["stopping"]])
    min_times = _parse_floats(min_time_grid, [float(base_cfg["decoder"]["min_time_s"])])
    target_accuracies = _parse_floats(
        target_accuracy_grid,
        [float(base_cfg["decoder"]["target_accuracy"])],
    )

    rows = []
    total = (
        len(passbands)
        * len(tmins)
        * len(encodings)
        * len(stoppings)
        * len(min_times)
        * len(target_accuracies)
    )
    logger.info(f"Running {total} offline decoder experiment(s).")

    for i, (passband, tmin, encoding, stopping, min_time, target_accuracy) in enumerate(
        product(passbands, tmins, encodings, stoppings, min_times, target_accuracies),
        start=1,
    ):
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["passband_hz"] = [float(passband[0]), float(passband[1])]
        cfg["decoder"]["tmin_s"] = float(tmin)
        cfg["decoder"]["encoding_length_s"] = float(encoding)
        cfg["decoder"]["stopping"] = stopping
        cfg["decoder"]["min_time_s"] = float(min_time)
        cfg["decoder"]["target_accuracy"] = float(target_accuracy)

        logger.info(
            f"[{i}/{total}] passband={passband}, tmin={tmin}, "
            f"encoding={encoding}, stopping={stopping}, min_time={min_time}, "
            f"target_accuracy={target_accuracy}"
        )

        cmeta = classifier_meta_from_cfg(cfg)
        X, y = _load_epochs_and_labels(cfg)
        V = _load_stimulus(cfg, use_full_codebook=use_full_codebook)
        _validate_labels_match_stimulus(y, V, use_full_codebook)
        acc, dur = calc_cv_accuracy_early_stop(cmeta, X, y, V, n_folds=n_folds)

        row = {
            "passband_low": passband[0],
            "passband_high": passband[1],
            "tmin_s": tmin,
            "encoding_length_s": encoding,
            "stopping": stopping,
            "min_time_s": min_time,
            "target_accuracy": target_accuracy,
            "n_trials": int(X.shape[0]),
            "n_classes": int(len(np.unique(y))),
            "n_folds": int(n_folds),
            "accuracy_mean": float(np.mean(acc)),
            "accuracy_std": float(np.std(acc)),
            "duration_mean": float(np.mean(dur)),
            "duration_std": float(np.std(dur)),
            "fold_accuracy": ";".join(f"{v:.6f}" for v in acc),
            "fold_duration": ";".join(f"{v:.6f}" for v in dur),
        }
        rows.append(row)

        logger.info(
            f"accuracy={row['accuracy_mean']:.3f} +/- {row['accuracy_std']:.3f}, "
            f"duration={row['duration_mean']:.2f}s +/- {row['duration_std']:.2f}s"
        )

    rows = sorted(rows, key=lambda r: (-r["accuracy_mean"], r["duration_mean"]))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    logger.info(
        "Best setting: "
        f"accuracy={best['accuracy_mean']:.3f}, duration={best['duration_mean']:.2f}s, "
        f"passband=[{best['passband_low']}, {best['passband_high']}], "
        f"tmin={best['tmin_s']}, encoding={best['encoding_length_s']}, "
        f"stopping={best['stopping']}"
    )
    logger.info(f"Saved offline results to {output_csv}")

    return rows


if __name__ == "__main__":
    Fire(run_grid)
