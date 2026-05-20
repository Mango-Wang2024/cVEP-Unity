# Mockup Streamer

A lightweight LSL mockup streamer for [Dareplane](https://github.com/bsdlab/Dareplane) with fast startup times.

## Features

- **Fast startup** (~0.2s) for random data streaming
- Lazy loading of heavy dependencies (mne, scipy) for file-based streaming
- Support for random data generation with configurable markers
- Support for file-based streaming (BrainVision, MNE/FIF, XDF formats)

## Installation

```bash
# Basic installation (random streaming only)
pip install -e .

# With file format support (mne, pyxdf)
pip install -e ".[files]"

# With development dependencies
pip install -e ".[all]"
```

**Note**: Requires Python `>=3.10`. For Python < 3.11, install `tomli` for TOML parsing.

## Quick Start

### Random Data Streaming (Fast)

```bash
# CLI for random data
python -m mockup_streamer.random_cli --help
python -m mockup_streamer.random_cli --n_channels=10 --sfreq=100

# Or via TCP server (fast startup)
python -m api.server
# Then connect and send: START_RANDOM
```

### File-based Streaming

```bash
# Configure streams in ./config/streams.toml, then:
python -m mockup_streamer.main

# Or via TCP server
python -m api.server
# Then connect and send: START
```

## TCP Server

Start the server:

```bash
python -m api.server --port=8080 --ip=127.0.0.1
```

Connect via telnet:

```bash
telnet 127.0.0.1 8080
```

Commands:
- `START_RANDOM` - Start random data streaming (fast, no heavy imports)
- `START` - Start file-based streaming (loads mne/scipy on demand)
- `STOP` - Stop streaming

## Configuration

Edit `./config/streams.toml` for stream settings:

```toml
[random.stream1]
stream_name = 'mock_random1'
sampling_freq = 100
n_channels = 2
pre_buffer_s = 300

[random.stream1.markers]
marker_stream_name = 'mock_random1_markers'
t_interval_s = 1
values = ['a', 'b', 'c']
```

## Python API

```python
from mockup_streamer.random_streamer import run_random_stream_thread

# Start streaming in background thread
thread, stop_event = run_random_stream_thread(
    n_channels=10,
    sfreq=100,
    stream_name="my_stream"
)

# Stop streaming
stop_event.set()
thread.join()
```

## Architecture

The module is structured for fast startup:

- `mockup_streamer/random_streamer.py` - Lightweight random streamer (no mne/scipy)
- `mockup_streamer/loaders/` - Lazy-loaded file format handlers
- `mockup_streamer/main.py` - Full-featured streamer with file support
- `api/server.py` - Dareplane TCP server interface
