# Expose a few parameters usually accessible via config as a CLI
import threading

from fire import Fire

from mockup_streamer.random_streamer import run_random_stream


def cli(
    n_channels: int = 10,
    sfreq: float = 100.0,
    pre_buffer_s: int = 300,
    stream_name: str = "mockup_random",
    markers_t_s: float = 1.0,
    marker_values: list = ["a", "b", "c"],
):
    """A simple CLI to spawn an LSL mockup stream with random data.

    Parameters
    ----------
    n_channels : int
        Number of channels.
    sfreq : float
        Sampling frequency in Hz.
    pre_buffer_s : int
        Seconds of data to pre-generate per buffer.
    stream_name : str
        Name of the LSL stream.
    markers_t_s : float
        Time interval of markers in seconds.
    marker_values : list
        Values of markers to cycle through.
    """
    stop_event = threading.Event()
    run_random_stream(
        stop_event=stop_event,
        n_channels=n_channels,
        sfreq=sfreq,
        pre_buffer_s=pre_buffer_s,
        stream_name=stream_name,
        markers_t_s=markers_t_s,
        marker_values=marker_values,
    )


if __name__ == "__main__":
    Fire(cli)
