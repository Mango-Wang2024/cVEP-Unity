"""Dareplane mockup streamer server.

Uses lazy imports to minimize startup time for random streaming.
"""
from functools import partial

from dareplane_utils.default_server.server import DefaultServer
from fire import Fire

from mockup_streamer.random_streamer import run_random_stream_thread
from mockup_streamer.utils.logging import logger


def _run_file_stream_thread(**kwargs):
    """Lazy import for file-based streaming (loads mne/scipy on demand)."""
    from mockup_streamer.main import run_mockup_streamer_thread

    return run_mockup_streamer_thread(random_data=False, **kwargs)


def run_server(port: int = 8080, ip: str = "127.0.0.1", loglevel: int = 10, **kwargs):
    """Start the mockup streamer server.

    Parameters
    ----------
    port : int
        Server port (default: 8080).
    ip : str
        Server IP address (default: 127.0.0.1).
    loglevel : int
        Logging level (default: 10/DEBUG).
    **kwargs
        Additional arguments passed to stream functions.
    """
    logger.setLevel(loglevel)

    pcommand_map = {
        "START": partial(_run_file_stream_thread, **kwargs),
        "START_RANDOM": partial(run_random_stream_thread, **kwargs),
    }

    logger.debug("Initializing server")
    server = DefaultServer(
        port=port, ip=ip, pcommand_map=pcommand_map, name="mockup_server"
    )

    server.init_server()

    logger.debug("Starting to listen on server")
    server.start_listening()
    logger.debug("Stopped listening on server")

    return 0


if __name__ == "__main__":
    Fire(run_server)
