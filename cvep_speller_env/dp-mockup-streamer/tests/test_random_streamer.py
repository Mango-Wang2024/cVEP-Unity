"""Tests for the lightweight random streamer module."""
import threading
import time

import numpy as np
import pylsl
import pytest

from mockup_streamer.random_streamer import (
    RandomStreamConfig,
    RandomStreamer,
    run_random_stream,
    run_random_stream_thread,
)


class TestRandomStreamConfig:
    def test_default_config(self):
        cfg = RandomStreamConfig()
        assert cfg.stream_name == "mockup_random"
        assert cfg.n_channels == 10
        assert cfg.sampling_freq == 100.0
        assert cfg.pre_buffer_s == 300
        assert cfg.stream_type == "EEG"
        assert cfg.markers is None

    def test_custom_config(self):
        cfg = RandomStreamConfig(
            stream_name="test_stream",
            n_channels=5,
            sampling_freq=250.0,
            markers={"t_interval_s": 2, "values": ["x", "y"]},
        )
        assert cfg.stream_name == "test_stream"
        assert cfg.n_channels == 5
        assert cfg.sampling_freq == 250.0
        assert cfg.markers == {"t_interval_s": 2, "values": ["x", "y"]}


class TestRandomStreamer:
    def test_streamer_initialization(self):
        cfg = RandomStreamConfig(
            stream_name="test_init",
            n_channels=3,
            sampling_freq=100,
            pre_buffer_s=10,
        )
        streamer = RandomStreamer(cfg)

        assert streamer.buffer.shape == (1000, 3)
        assert streamer.buffer.dtype == np.float32
        assert streamer.outlet is not None
        assert streamer.outlet_mrk is None
        assert streamer.buffer_i == 0
        assert streamer.n_pushed == 0

    def test_streamer_with_markers(self):
        cfg = RandomStreamConfig(
            stream_name="test_markers",
            n_channels=2,
            sampling_freq=100,
            pre_buffer_s=10,
            markers={"t_interval_s": 1, "values": ["a", "b", "c"]},
        )
        streamer = RandomStreamer(cfg)

        assert streamer.markers is not None
        assert streamer.markers.shape == (10, 2)
        assert streamer.outlet_mrk is not None
        # Check marker cycling
        assert streamer.markers[0, 1] == "a"
        assert streamer.markers[1, 1] == "b"
        assert streamer.markers[2, 1] == "c"
        assert streamer.markers[3, 1] == "a"

    def test_push_updates_indices(self):
        cfg = RandomStreamConfig(
            stream_name="test_push",
            n_channels=2,
            sampling_freq=100,
            pre_buffer_s=30,
        )
        streamer = RandomStreamer(cfg)

        # Reset start time and wait a bit
        streamer.t_start_s = pylsl.local_clock()
        time.sleep(0.15)
        streamer.push()

        assert streamer.buffer_i > 0
        assert streamer.n_pushed > 0
        assert streamer.buffer_i == streamer.n_pushed

    def test_buffer_regeneration(self):
        cfg = RandomStreamConfig(
            stream_name="test_regen",
            n_channels=2,
            sampling_freq=100,
            pre_buffer_s=1,  # Small buffer to trigger regeneration
        )
        streamer = RandomStreamer(cfg)

        # Reset and push enough to exceed buffer
        streamer.t_start_s = pylsl.local_clock()
        time.sleep(1.2)
        streamer.push()

        # Buffer should have been regenerated
        assert streamer.buffer_i == 0
        assert streamer.n_pushed == 0


class TestRunRandomStream:
    def test_thread_function(self):
        thread, stop_event = run_random_stream_thread(
            n_channels=2,
            sfreq=100,
            pre_buffer_s=5,
            stream_name="test_thread",
        )

        assert thread.is_alive()
        time.sleep(0.3)
        stop_event.set()
        thread.join(timeout=2)
        assert not thread.is_alive()


class TestStreamingIntegration:
    def test_sender_receiver(self):
        """Test that data can be received via LSL."""
        cfg = RandomStreamConfig(
            stream_name="test_sr_random",
            n_channels=3,
            sampling_freq=200,
            pre_buffer_s=30,
            markers={
                "marker_stream_name": "test_sr_random_mrk",
                "t_interval_s": 1,
                "values": ["a", "b", "c"],
            },
        )
        streamer = RandomStreamer(cfg)
        buffer_data = []

        # Find the stream
        streams = pylsl.resolve_streams()
        idx_test = next(
            i for i, s in enumerate(streams) if s.name() == "test_sr_random"
        )

        # Start listener
        stop_event = threading.Event()
        inlet = pylsl.StreamInlet(streams[idx_test])

        def listener():
            while not stop_event.is_set():
                time.sleep(0.05)
                chunk, _ = inlet.pull_chunk()
                if chunk:
                    buffer_data.extend(chunk)

        listener_thread = threading.Thread(target=listener)
        listener_thread.start()

        # Push data
        streamer.t_start_s = pylsl.local_clock()
        for _ in range(20):
            time.sleep(0.1)
            streamer.push()

        stop_event.set()
        listener_thread.join(timeout=2)

        # Verify received data
        assert len(buffer_data) > 0
        assert len(buffer_data[0]) == 3  # n_channels
