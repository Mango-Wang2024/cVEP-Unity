"""Lightweight random data streamer without heavy dependencies.

This module provides fast random data streaming without importing mne/scipy.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pylsl

from mockup_streamer.utils.logging import logger


@dataclass
class RandomStreamConfig:
    """Configuration for a random data stream."""

    stream_name: str = "mockup_random"
    n_channels: int = 10
    sampling_freq: float = 100.0
    pre_buffer_s: int = 300
    stream_type: str = "EEG"
    markers: dict | None = None


@dataclass
class RandomStreamer:
    """Lightweight random data streamer.

    Attributes
    ----------
    cfg : RandomStreamConfig
        Stream configuration.
    outlet : pylsl.StreamOutlet
        LSL outlet for data.
    outlet_mrk : pylsl.StreamOutlet | None
        LSL outlet for markers.
    """

    cfg: RandomStreamConfig
    buffer: np.ndarray = field(init=False, repr=False)
    markers: np.ndarray | None = field(init=False, default=None)
    buffer_i: int = field(init=False, default=0)
    n_pushed: int = field(init=False, default=0)
    t_start_s: float = field(init=False, default=0.0)
    outlet: pylsl.StreamOutlet = field(init=False, repr=False)
    outlet_mrk: pylsl.StreamOutlet | None = field(init=False, default=None)

    def __post_init__(self):
        self._generate_data()
        self._init_outlet()
        if self.cfg.markers:
            self._init_marker_outlet()

    def _generate_data(self):
        """Generate random data buffer."""
        n_samples = int(self.cfg.sampling_freq * self.cfg.pre_buffer_s)
        self.buffer = np.random.randn(n_samples, self.cfg.n_channels).astype(np.float32)
        self.buffer_i = 0
        self.n_pushed = 0
        self.t_start_s = pylsl.local_clock()

        if self.cfg.markers:
            self._generate_markers(n_samples)

    def _generate_markers(self, n_samples: int):
        """Generate marker array."""
        dt = self.cfg.markers.get("t_interval_s", 1)
        nmrk = int(self.cfg.pre_buffer_s / dt)
        self.markers = np.zeros((nmrk, 2), dtype=object)
        self.markers[:, 0] = np.arange(nmrk) * dt * self.cfg.sampling_freq

        mval = self.cfg.markers.get("values", ["a"])
        mval = mval if isinstance(mval, list) else [mval]
        mvals = np.tile(mval, nmrk // len(mval) + 1)
        self.markers[:, 1] = mvals[:nmrk]

    def _init_outlet(self):
        """Initialize LSL data outlet."""
        info = pylsl.StreamInfo(
            self.cfg.stream_name,
            self.cfg.stream_type,
            self.cfg.n_channels,
            self.cfg.sampling_freq,
            pylsl.cf_float32,
        )
        self.outlet = pylsl.StreamOutlet(info)

    def _init_marker_outlet(self):
        """Initialize LSL marker outlet."""
        name = self.cfg.markers.get(
            "marker_stream_name", f"{self.cfg.stream_name}_markers"
        )
        info = pylsl.StreamInfo(name, "Markers", 1, pylsl.IRREGULAR_RATE, "string")
        self.outlet_mrk = pylsl.StreamOutlet(info)

    def push(self):
        """Push required samples based on elapsed time."""
        n_required = (
            int((pylsl.local_clock() - self.t_start_s) * self.cfg.sampling_freq)
            - self.n_pushed
        )
        if n_required <= 0:
            return

        data = self.buffer[self.buffer_i : self.buffer_i + n_required]
        self.outlet.push_chunk(data.tolist())

        if self.outlet_mrk is not None and self.markers is not None:
            self._push_markers(self.n_pushed, self.n_pushed + n_required)

        self.buffer_i += n_required
        self.n_pushed += n_required

        if self.buffer_i >= self.buffer.shape[0]:
            logger.debug("Regenerating random buffer")
            self._generate_data()

    def _push_markers(self, idx_from: int, idx_to: int):
        """Push markers within index range."""
        msk = (self.markers[:, 0] >= idx_from) & (self.markers[:, 0] < idx_to)
        for mrk in self.markers[msk, 1]:
            self.outlet_mrk.push_sample([str(mrk)])


def run_random_stream(
    stop_event: threading.Event,
    n_channels: int = 10,
    sfreq: float = 100.0,
    pre_buffer_s: int = 300,
    stream_name: str = "mockup_random",
    stream_type: str = "EEG",
    markers_t_s: float = 1.0,
    marker_values: list | None = None,
) -> int:
    """Run a random data stream.

    Parameters
    ----------
    stop_event : threading.Event
        Event to signal stream stop.
    n_channels : int
        Number of channels.
    sfreq : float
        Sampling frequency in Hz.
    pre_buffer_s : int
        Buffer size in seconds.
    stream_name : str
        Name of the LSL stream.
    stream_type : str
        Type of the LSL stream (e.g., "EEG").
    markers_t_s : float
        Marker interval in seconds.
    marker_values : list | None
        Marker values to cycle through.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    marker_values = marker_values or ["a", "b", "c"]

    cfg = RandomStreamConfig(
        stream_name=stream_name,
        n_channels=n_channels,
        sampling_freq=sfreq,
        pre_buffer_s=pre_buffer_s,
        stream_type=stream_type,
        markers={"t_interval_s": markers_t_s, "values": marker_values},
    )

    streamer = RandomStreamer(cfg)
    dt = 1 / sfreq

    logger.info(f"Starting random stream: {stream_name}")

    while not stop_event.is_set():
        time.sleep(dt)
        streamer.push()

    return 0


def run_random_stream_thread(
    **kwargs,
) -> tuple[threading.Thread, threading.Event]:
    """Run random streaming in a separate thread.

    Returns
    -------
    tuple[threading.Thread, threading.Event]
        The thread and stop event.
    """
    stop_event = threading.Event()
    thread = threading.Thread(
        target=run_random_stream, kwargs={"stop_event": stop_event, **kwargs}
    )
    thread.start()
    return thread, stop_event
