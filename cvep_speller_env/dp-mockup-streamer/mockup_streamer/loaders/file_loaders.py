"""File loaders for different data formats.

These are separated to enable lazy importing of heavy dependencies like mne.
"""
from pathlib import Path

import numpy as np


def load_data(fp: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Load data from file based on suffix."""
    loaders = {
        ".vhdr": load_bv,
        ".fif": load_mne,
        ".xdf": load_xdf,
    }
    data, markers, sfreq = loaders[fp.suffix](fp, cfg)
    markers = None if len(markers) == 0 else markers
    return data, markers, sfreq


def load_bv(fp: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Load BrainVision files."""
    import mne

    raw = mne.io.read_raw_brainvision(fp, preload=True)
    data, markers = mne_raw_to_data_and_markers(raw, cfg)
    return data, markers, raw.info["sfreq"]


def load_mne(fp: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Load mne/fif files."""
    import mne

    raw = mne.io.read_raw(fp, preload=True)
    data, markers = mne_raw_to_data_and_markers(raw, cfg)
    return data, markers, raw.info["sfreq"]


def load_xdf(fp: Path, cfg: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """Load from xdf file.

    Parameters
    ----------
    fp : Path
        Path to xdf file.
    cfg : dict
        Configuration dictionary with stream_name, optional markers config,
        and pyxdf_kwargs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        Data, markers, and sampling frequency.
    """
    import pyxdf

    marker_stream = cfg.get("markers", {}).get("marker_stream_name", "")
    d = pyxdf.load_xdf(fp, **cfg.get("pyxdf_kwargs", {}))
    snames = [s["info"]["name"][0] for s in d[0]]

    try:
        sdata = d[0][snames.index(cfg["stream_name"])]
    except ValueError:
        raise KeyError(f"Stream {cfg['stream_name']=} not found in {snames=}")

    sfreq = float(sdata["info"]["nominal_srate"][0])
    data = sdata["time_series"]

    if marker_stream != "":
        td = sdata["time_stamps"]
        mdata = d[0][snames.index(marker_stream)]
        tm = mdata["time_stamps"]
        idx = [np.argmin(np.abs(td - t)) for t in tm]
        markers = np.asarray(
            [idx, [v[0] for v in mdata["time_series"]]], dtype="object"
        ).T
    else:
        markers = np.asarray([])

    if isinstance(data, list):
        data = np.array(data)

    return data, markers, sfreq


def mne_raw_to_data_and_markers(raw, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert mne Raw to data array and markers."""
    import mne

    if cfg.get("select_channels", "") != "":
        raw.pick_channels(cfg["select_channels"])

    ev, evid = mne.events_from_annotations(raw, verbose=False)
    imap = {v: k for k, v in evid.items()}
    ev = ev.astype("object")
    ev[:, 2] = [imap[i] for i in ev[:, 2]]

    data = raw.get_data().T
    return data, ev[:, [0, 2]]
