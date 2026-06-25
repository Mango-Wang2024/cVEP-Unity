from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import filtfilt, iirnotch, lfilter, lfilter_zi


DEFAULT_LINE_NOISE_HZ = (50.0,)


def _valid_notch_freqs(sfreq: float, freqs: tuple[float, ...] | list[float]) -> list[float]:
    nyquist = float(sfreq) / 2.0
    return [float(freq) for freq in freqs if 0.0 < float(freq) < nyquist]


class StatefulNotchFilter:
    """Small streaming IIR notch filter for power-line interference."""

    def __init__(
            self,
            sfreq: float,
            freqs: tuple[float, ...] | list[float] = DEFAULT_LINE_NOISE_HZ,
            quality: float = 30.0,
    ):
        self.sfreq = float(sfreq)
        self.filters = [
            iirnotch(w0=freq, Q=float(quality), fs=self.sfreq)
            for freq in _valid_notch_freqs(self.sfreq, freqs)
        ]
        self.zi: list[NDArray | None] = [None for _ in self.filters]

    @property
    def enabled(self) -> bool:
        return len(self.filters) > 0

    def filter(self, samples: NDArray) -> NDArray:
        x = np.asarray(samples)
        if x.size == 0 or not self.filters:
            return x

        y = x
        for i, (b, a) in enumerate(self.filters):
            if self.zi[i] is None or self.zi[i].shape[1] != y.shape[1]:
                self.zi[i] = lfilter_zi(b, a)[:, None] * y[0][None, :]
            y, self.zi[i] = lfilter(b, a, y, axis=0, zi=self.zi[i])
        return y


def apply_notch_filter(
        samples: NDArray,
        sfreq: float,
        freqs: tuple[float, ...] | list[float] = DEFAULT_LINE_NOISE_HZ,
        quality: float = 30.0,
) -> NDArray:
    """Apply a zero-phase notch filter to a complete EEG array."""

    x = np.asarray(samples)
    if x.size == 0:
        return x

    y = x
    for freq in _valid_notch_freqs(float(sfreq), freqs):
        b, a = iirnotch(w0=freq, Q=float(quality), fs=float(sfreq))
        padlen = 3 * max(len(a), len(b))
        if y.shape[0] > padlen:
            y = filtfilt(b, a, y, axis=0)
        else:
            y = lfilter(b, a, y, axis=0)
    return y
