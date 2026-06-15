import json
import queue
import socket
import threading
import time
from pathlib import Path

import joblib
import numpy as np
import pylsl
import toml
from dareplane_utils.general.time import sleep_s
from dareplane_utils.logging.logger import get_logger
from dareplane_utils.signal_processing.filtering import FilterBank
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from fire import Fire
from numpy.typing import NDArray
from scipy.signal import resample

from cvep_decoder.utils.logging import logger


REALTIME_UDP_FIX_VERSION = "udp-arm-v11-auto-decision-timer"


class OnlineDecoder:
    """Decoder class to evaluate a classifier based on data from an LSL stream.

    Parameters
    ----------
    decoder_file : Path
        Path to the classifier object to use for decoding.

    decoder_meta_file : Path
        Path to the dictionary with metadata about the classifier.

    marker_stream_name : str
        Name of the marker stream to read the start_eval_marker from which decoding will be started.

    data_stream_name : str
        Name of the input stream containing the signal used for decoding.

    decoder_stream_name : str
        Name of the output stream to which the result of classifier.predict() are written.

    buffer_size_s : float
        Defines the size of the buffer for the data and marker stream.

    start_eval_marker : str
        The marker that starts the continuous evaluation.

    max_eval_time_s : float
        The maximum time to try decoding a trial. Default is 10s.

    first_trial_max_eval_time_s : float | None
        Optional longer maximum evaluation time for the first online trial.

    t_sleep_s : float
        Time to sleep between updates, defines the update frequency. Default is 0.1s.

    selected_channels : list[str] | list[int] | None
        If a list of channel names is provided, data of only those channels will be processed. Default is None, which
         means all channels are considered. If a list of integers is provided, they are interpreted as indices.
    """

    def __init__(
            self,
            decoder_file: Path,
            decoder_meta_file: Path,
            marker_stream_name: str,
            marker_udp_host: str,
            marker_udp_port: int | None,
            decoder_output_udp_host: str,
            decoder_output_udp_port: int | None,
            data_stream_name: str,
            decoder_stream_name: str,
            buffer_size_s: float,
            padding_size_s: float,
            start_eval_marker: str,
            max_eval_time_s: float = 10,
            first_trial_max_eval_time_s: float | None = None,
            t_sleep_s: float = 0.1,
            selected_channels: list[str] | None = None,
            n_positions: int = 0,
            soft_decision_enabled: bool = False,
            soft_decision_smoothing: float = 0.5,
            soft_decision_min_confidence: float = 0.0,
            soft_decision_temperature: float = 1.0,
    ):

        self.classifier_path = decoder_file
        self.classifier_meta_path = decoder_meta_file
        self.marker_stream_name = marker_stream_name
        self.marker_udp_host = marker_udp_host
        self.marker_udp_port = marker_udp_port
        self.decoder_output_udp_host = decoder_output_udp_host
        self.decoder_output_udp_port = decoder_output_udp_port
        self.data_stream_name = data_stream_name
        self.decoder_stream_name = decoder_stream_name
        self.buffer_size_s = buffer_size_s
        self.padding_size_s = padding_size_s
        self.start_eval_marker = start_eval_marker
        self.max_eval_time_s = max_eval_time_s
        self.first_trial_max_eval_time_s = first_trial_max_eval_time_s
        self.t_sleep_s = t_sleep_s
        self.selected_channels = selected_channels
        self.n_positions = n_positions
        self.soft_decision_enabled = soft_decision_enabled
        self.soft_decision_smoothing = soft_decision_smoothing
        self.soft_decision_min_confidence = soft_decision_min_confidence
        self.soft_decision_temperature = soft_decision_temperature

        self.selected_ch_idx = None
        self.is_decoding: bool = False
        self.start_eval_time: float = 0.0
        self.internal_decoding_start_time: float = time.time()
        self.current_trial_id: int | None = None
        self.classifier = None
        self.classifier_meta = None

        self.input_mrk_sw: StreamWatcher | None = None
        self.marker_inlet: pylsl.StreamInlet | None = None
        self.marker_udp_socket: socket.socket | None = None
        self.marker_udp_thread: threading.Thread | None = None
        self.marker_udp_worker_thread: threading.Thread | None = None
        self.marker_udp_stop_event: threading.Event | None = None
        self.marker_udp_queue: queue.Queue[tuple[str, float]] | None = None
        self.decoder_output_udp_socket: socket.socket | None = None
        self.marker_time_correction_s: float = 0.0
        self.marker_time_correction_valid: bool = False
        self.input_sw: StreamWatcher | None = None
        self.input_sfreq: int | None = None
        self.input_chs_info: list[dict[str, str]] | None = None
        self.filterbank: FilterBank | None = None
        self.output_sw: StreamWatcher | None = None
        self.band = None
        self.classifier_input_sfreq = None
        self.last_soft_position: NDArray | None = None
        self.last_insufficient_log_time: float = 0.0
        self.max_marker_drain_samples: int = 10000
        self.max_marker_age_s: float = 1.0
        self.fallback_marker_delay_s: float = 0.2
        self.force_decision_marker: str = "force_decision"
        self.force_decision_requested: bool = False
        self.udp_decoding_enabled: bool = False
        self.completed_udp_trial_ids: set[int] = set()
        self.max_udp_packet_age_s: float = 2.0
        self.run_thread: threading.Thread | None = None
        self.run_stop_event: threading.Event | None = None
        self.eeg_update_thread: threading.Thread | None = None
        self.eeg_update_stop_event: threading.Event | None = None
        self.udp_trial_timer: threading.Timer | None = None
        self.run_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.eeg_update_interval_s = min(max(self.t_sleep_s, 0.005), 0.02)
        if self.decoder_output_udp_port is not None:
            self.decoder_output_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _max_eval_time_for_trial(self, trial_id: int | None = None) -> float:
        if trial_id is None:
            trial_id = self.current_trial_id

        if (
                trial_id == 1
                and self.first_trial_max_eval_time_s is not None
                and self.first_trial_max_eval_time_s > 0
        ):
            return float(self.first_trial_max_eval_time_s)

        return float(self.max_eval_time_s)

    # -------- Connection and initialization methods --------------------------
    def load_model(
            self,
            classifier_path: Path | None = None,
            classifier_meta_path: Path | None = None,
    ) -> int:
        """Loading the model and allowing for overwrites"""

        cp = (
            classifier_path
            if classifier_path is not None
            else self.classifier_path
        )
        cmp = (
            classifier_meta_path
            if classifier_meta_path is not None
            else self.classifier_meta_path
        )
        logger.info(f"Loading classifier from {cp=} and {cmp=}.")
        logger.info(f"[CHECK] LOAD MODEL started: decoder model={cp}, meta={cmp}.")

        try:
            self.classifier = joblib.load(cp)
            self.classifier_meta = json.load(open(cmp, "r"))
        except FileNotFoundError:
            logger.error(f"Could not load classifier from {cp=} or {cmp=}. Validate that both exist.")
            logger.error(
                f"[CHECK] LOAD MODEL failed: missing decoder model or meta file. "
                f"Expected model={cp}, meta={cmp}."
            )
            return 1

        self.classifier_input_sfreq = self.classifier_meta["sfreq"]
        self.band = self.classifier_meta["fband"]
        logger.info(
            f"[CHECK] LOAD MODEL finished: model is ready "
            f"(classifier_sfreq={self.classifier_input_sfreq} Hz, band={self.band})."
        )

        return 0

    def connect_marker_stream(self):
        logger.info(f'Connecting to marker stream "{self.marker_stream_name}".')
        streams = []
        while len(streams) == 0:
            streams = pylsl.resolve_byprop("name", self.marker_stream_name, timeout=1)
            if len(streams) == 0:
                logger.info(f'Waiting for marker stream "{self.marker_stream_name}".')

        if len(streams) > 1:
            logger.warning(f'Selecting first marker stream named "{self.marker_stream_name}".')

        if streams[0].channel_count() != 1:
            logger.error("The marker stream should have exactly one channel.")
            return 1

        self.marker_inlet = pylsl.StreamInlet(
            streams[0],
            max_buflen=1,
            max_chunklen=1,
            recover=True,
        )
        try:
            self.marker_time_correction_s = self.marker_inlet.time_correction(timeout=1.0)
            self.marker_time_correction_valid = np.isfinite(self.marker_time_correction_s)
        except Exception as err:
            logger.warning(f"Could not estimate marker stream time correction: {err}")
            self.marker_time_correction_s = 0.0
            self.marker_time_correction_valid = False
        logger.info(
            f"[CHECK] Marker stream time correction is {self.marker_time_correction_s:.6f}s "
            f"(valid={self.marker_time_correction_valid})."
        )

        return 0

    def start_udp_marker_listener(self, restart: bool = False) -> None:
        if self.marker_udp_port is None:
            logger.info("[CHECK] UDP marker listener disabled.")
            return
        if self.marker_udp_socket is not None:
            if not restart:
                return
            self.stop_udp_marker_listener()

        self.marker_udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.marker_udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.marker_udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1_048_576)
        try:
            self.marker_udp_socket.bind((self.marker_udp_host, self.marker_udp_port))
        except OSError as err:
            self.marker_udp_socket.close()
            self.marker_udp_socket = None
            logger.error(
                f"[CHECK] Could not start UDP marker listener on "
                f"{self.marker_udp_host}:{self.marker_udp_port}: {err}. "
                "Close old control-room/decoder Python processes and restart."
            )
            raise
        self.marker_udp_socket.settimeout(0.1)
        self.marker_udp_queue = queue.Queue()
        self.marker_udp_stop_event = threading.Event()
        self.marker_udp_thread = threading.Thread(
            target=self._udp_marker_loop,
            kwargs={"stop_event": self.marker_udp_stop_event},
            daemon=True,
        )
        self.marker_udp_worker_thread = threading.Thread(
            target=self._udp_marker_worker_loop,
            kwargs={"stop_event": self.marker_udp_stop_event},
            daemon=True,
        )
        self.marker_udp_thread.start()
        self.marker_udp_worker_thread.start()
        logger.info(
            f"[CHECK] UDP marker listener started on "
            f"{self.marker_udp_host}:{self.marker_udp_port}."
        )
        logger.info("[CHECK] UDP marker worker started.")

    def stop_udp_marker_listener(self) -> None:
        if self.marker_udp_stop_event is not None:
            self.marker_udp_stop_event.set()
        if self.marker_udp_socket is not None:
            try:
                self.marker_udp_socket.close()
            except OSError:
                pass

        current_thread = threading.current_thread()
        for thread in (self.marker_udp_thread, self.marker_udp_worker_thread):
            if thread is not None and thread is not current_thread and thread.is_alive():
                thread.join(timeout=0.2)

        self.marker_udp_socket = None
        self.marker_udp_thread = None
        self.marker_udp_worker_thread = None
        self.marker_udp_stop_event = None
        self.marker_udp_queue = None

    def connect_data_stream(self):
        logger.info(f'Connecting to data stream "{self.data_stream_name}".')
        logger.info(
            f'[CHECK] CONNECT DECODER waiting for EEG stream "{self.data_stream_name}". '
            "If this line stays visible for a long time, check OpenBCI LSL and Lab Recorder Update."
        )
        self.input_sw = StreamWatcher(self.data_stream_name, buffer_size_s=self.buffer_size_s, logger=logger)
        self.input_sw.connect_to_stream()

        self.input_sfreq = int(self.input_sw.inlet.info().nominal_srate())
        self.input_chs_info = [dict(ch_name=ch_name, type="EEG") for ch_name in self.input_sw.channel_names]

        if self.selected_channels is None:
            self.selected_ch_idx = list(range(len(self.input_chs_info)))
        else:
            if isinstance(self.selected_channels[0], str):
                self.selected_ch_idx = [
                    self.input_sw.channel_names.index(ch)
                    for ch in self.selected_channels
                ]
            elif isinstance(self.selected_channels[0], int):
                self.selected_ch_idx = self.selected_channels
            else:
                raise logger.error(f"{self.selected_channels=} must be a list of `str` or `int` or `None`.")
            return 1

        return 0

    def create_filterbank(self):
        logger.info("Creating the classifier and the filter bank.")
        assert self.input_chs_info, (
            f"self.input_chs_info is {self.input_chs_info}. Please connect to "
            "the input lsl stream to derive channel information by calling "
            " `self.connect_data_stream()`"
        )

        self.filterbank = FilterBank(
            bands={"band": self.band},
            sfreq=self.input_sfreq,
            output="signal",
            n_in_channels=len(self.selected_ch_idx),
            filter_buffer_s=self.buffer_size_s,
        )

    def create_decoder_stream(self):
        logger.info(f'Creating the decoder stream "{self.decoder_stream_name}"')
        info = pylsl.StreamInfo(
            self.decoder_stream_name,
            type="MISC",
            channel_count=2 if self.soft_decision_enabled else 1,
            nominal_srate=pylsl.IRREGULAR_RATE,
            channel_format=pylsl.cf_float32 if self.soft_decision_enabled else pylsl.cf_int8,
            source_id="decoder_stream_id",
        )

        if self.soft_decision_enabled:
            channels = info.desc().append_child("channels")
            for label in ["x", "y"]:
                channels.append_child("channel").append_child_value("label", label)

        self.output_sw = pylsl.StreamOutlet(info)

    def start_eeg_update_thread(self) -> None:
        if self.input_sw is None or self.filterbank is None:
            logger.error("[CHECK] Cannot start EEG update thread before EEG stream/filterbank exist.")
            return
        if self.eeg_update_thread is not None and self.eeg_update_thread.is_alive():
            return

        self.eeg_update_stop_event = threading.Event()
        self.eeg_update_thread = threading.Thread(
            target=self._eeg_update_loop,
            kwargs={"stop_event": self.eeg_update_stop_event},
            daemon=True,
        )
        self.eeg_update_thread.start()
        logger.info(
            f"[CHECK] EEG update thread started with interval "
            f"{self.eeg_update_interval_s:.3f}s."
        )

    def _eeg_update_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            t_start = time.perf_counter()
            with self.processing_lock:
                self._update_eeg_filter_once()

            elapsed_s = time.perf_counter() - t_start
            time.sleep(max(0.0, self.eeg_update_interval_s - elapsed_s))

    def init_all(self) -> int:
        """Return an int as this is exposed as a PCOMM `CONNECT_DECODER`"""
        logger.info("[CHECK] CONNECT DECODER pressed: initializing decoder streams.")
        self.connect_data_stream()
        logger.info(
            f'[CHECK] Connected EEG stream "{self.data_stream_name}" '
            f"with sfreq={self.input_sfreq} Hz and channels={self.input_sw.channel_names}."
        )
        self.flush_input_stream(reason="before online decoding setup")
        self.create_filterbank()
        logger.info("[CHECK] Filter bank created.")
        if self.marker_udp_port is not None:
            logger.info(
                "[CHECK] UDP mode active: marker listener has priority; decoder "
                "schedules an automatic rCCA decision from start_trial timing."
            )
        else:
            self.start_eeg_update_thread()
        self.create_decoder_stream()
        logger.info(
            f'[CHECK] Decoder output stream "{self.decoder_stream_name}" created '
            f"with soft_decision_enabled={self.soft_decision_enabled}."
        )
        if self.marker_udp_port is not None:
            self.udp_decoding_enabled = False
            logger.info(
                f"[CHECK] Real-time UDP fix active: {REALTIME_UDP_FIX_VERSION}."
            )
            logger.info(
                "[CHECK] UDP marker mode configured; fresh marker listener will "
                "start when DECODE ONLINE is pressed."
            )
        else:
            self.connect_marker_stream()
            logger.info(f'[CHECK] Connected marker stream "{self.marker_stream_name}".')
        return 0

    # ------------------ Online functionality ---------------------------------

    def run(self) -> tuple[threading.Thread, threading.Event]:
        with self.run_lock:
            if self.marker_udp_port is not None:
                logger.info(
                    "[CHECK] DECODE ONLINE pressed: decoder run loop starting "
                    "(starting fresh UDP marker listener)."
                )
                if self.udp_trial_timer is not None:
                    self.udp_trial_timer.cancel()
                    self.udp_trial_timer = None
                self.start_udp_marker_listener(restart=True)
                self.drain_udp_marker_queue(reason="before UDP decoder")
                with self.processing_lock:
                    self.flush_input_stream(reason="before UDP decoder")
                self.completed_udp_trial_ids.clear()
                self.is_decoding = False
                self.current_trial_id = None
                self.force_decision_requested = False
                self.start_eval_time = 0.0
                self.udp_decoding_enabled = True
                self.run_stop_event = threading.Event()
                return self.marker_udp_thread, self.run_stop_event

            if self.run_thread is not None and self.run_thread.is_alive():
                logger.info("[CHECK] DECODE ONLINE pressed but decoder run loop is already running.")
                return self.run_thread, self.run_stop_event

            logger.info("[CHECK] DECODE ONLINE pressed: decoder run loop starting.")
            self.drain_marker_stream(reason="before decoder run loop")
            self.flush_input_stream(reason="before decoder run loop")
            self.run_stop_event = threading.Event()
            self.run_thread = threading.Thread(
                target=self._run_loop,
                kwargs={"stop_event": self.run_stop_event},
                daemon=True,
            )
            self.run_thread.start()
            logger.debug("Started the run loop")
            return self.run_thread, self.run_stop_event

    def drain_marker_stream(self, reason: str = "") -> int:
        """Remove queued marker samples so decoding starts from fresh markers."""

        if self.marker_inlet is None:
            return 0

        n_drained = 0
        while n_drained < self.max_marker_drain_samples:
            sample, _ = self.marker_inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            n_drained += 1

        if n_drained > 0 or reason:
            suffix = f" {reason}" if reason else ""
            logger.info(f"[CHECK] Drained {n_drained} old marker samples{suffix}.")
        if n_drained >= self.max_marker_drain_samples:
            logger.warning(
                "[CHECK] Marker stream still had queued samples after draining "
                f"{self.max_marker_drain_samples}; marker production may be too fast."
            )

        return n_drained

    def flush_input_stream(self, reason: str = "") -> int:
        """Drop old EEG samples that arrived before online decoding started."""

        if self.input_sw is None or self.input_sw.inlet is None:
            return 0

        try:
            n_flushed = self.input_sw.inlet.flush()
        except Exception as err:
            logger.warning(f"[CHECK] Could not flush EEG inlet {reason}: {err}")
            return 0

        self.input_sw.n_new = 0
        if n_flushed > 0 or reason:
            suffix = f" {reason}" if reason else ""
            logger.info(f"[CHECK] Flushed {n_flushed} old EEG samples{suffix}.")

        return n_flushed

    def drain_udp_marker_socket(self, reason: str = "") -> int:
        """Remove queued UDP trial packets before starting a fresh online run."""

        if self.marker_udp_socket is None:
            return 0

        n_drained = 0
        while True:
            try:
                self.marker_udp_socket.recvfrom(2048)
            except BlockingIOError:
                break
            except OSError as err:
                logger.warning(f"[CHECK] Could not drain UDP marker socket {reason}: {err}")
                break
            n_drained += 1

        if n_drained > 0 or reason:
            suffix = f" {reason}" if reason else ""
            logger.info(f"[CHECK] Drained {n_drained} old UDP marker packet(s){suffix}.")

        return n_drained

    def drain_udp_marker_queue(self, reason: str = "") -> int:
        """Remove UDP packets already captured by the receiver thread."""

        if self.marker_udp_queue is None:
            return 0

        n_drained = 0
        while True:
            try:
                self.marker_udp_queue.get_nowait()
            except queue.Empty:
                break
            n_drained += 1

        if n_drained > 0 or reason:
            suffix = f" {reason}" if reason else ""
            logger.info(f"[CHECK] Drained {n_drained} queued UDP marker packet(s){suffix}.")

        return n_drained

    def update(self):
        if self.marker_udp_port is not None:
            # UDP markers are handled by the listener thread. Do not do heavy
            # continuous EEG/filter updates here, because that can starve the
            # UDP thread and make trial decisions arrive seconds too late.
            return

        # UDP is the low-latency online trigger. It can also request an
        # immediate best-score decision at the end of a trial.
        self.check_udp_marker_should_start()

        # Check markers before EEG/filter updates so the trial start is captured
        # with minimal latency even if the EEG update has accumulated backlog.
        if not self.is_decoding:
            self.check_if_decoding_should_start()

        self.input_sw.update()
        self._filter()

        # check if decoding should start after EEG update too, in case the
        # marker arrived while the inlet was empty before updating.
        if not self.is_decoding:
            self.check_if_decoding_should_start()
        else:
            elapsed_s = time.time() - self.internal_decoding_start_time
            max_eval_time_s = self._max_eval_time_for_trial()
            if self.force_decision_requested or elapsed_s > max_eval_time_s:
                reason = "force_decision request" if self.force_decision_requested else "timeout"
                if not self.force_decision_requested:
                    logger.info(f"Stopping decoding after max_eval_time_s={max_eval_time_s}")
                self.force_decision_requested = False
                x = self._create_epoch()
                if x.shape[2] > 0:
                    xs = self._resample(x)
                    if not self._force_classify(xs, reason=reason):
                        self.is_decoding = False
                        self.current_trial_id = None
                else:
                    logger.warning(f"[CHECK] Decoder {reason} reached with an empty EEG epoch; no output sent.")
                    self.is_decoding = False
                    self.current_trial_id = None
            else:
                # Decoding
                if self.is_decoding:
                    x = self._create_epoch()
                    if x.shape[2] > 0:
                        xs = self._resample(x)
                        self._classify(xs)

    def check_if_decoding_should_start(self):
        if self.check_udp_marker_should_start():
            return
        if self.marker_udp_port is not None:
            return

        if self.marker_inlet is None:
            logger.error("No marker stream connected, cannot start decoding based on markers.")
            return

        markers = []
        markers_t = []

        sample, timestamp = self.marker_inlet.pull_sample(timeout=self.t_sleep_s)
        if sample is None:
            return
        markers.append(sample[0])
        markers_t.append(timestamp)

        for _ in range(self.max_marker_drain_samples - 1):
            sample, timestamp = self.marker_inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            markers.append(sample[0])
            markers_t.append(timestamp)

        if len(markers) >= self.max_marker_drain_samples:
            logger.warning(
                "[CHECK] Marker update hit the drain limit. The decoder may still lag "
                "behind the marker stream."
            )

        if len(markers) > 0:
            markers = np.asarray(markers)
            markers_t = np.asarray(markers_t)

            if self.start_eval_marker in markers:
                # get time stamp of start_eval_marker --> consider inputs for epoch from this onwards
                idx = np.where(markers == self.start_eval_marker)[0]
                marker_lsl_time = float(markers_t[idx[-1]])
                eeg_times = self.input_sw.unfold_buffer_t()
                if len(eeg_times) == 0:
                    logger.warning(
                        f"[CHECK] Ignored marker '{self.start_eval_marker}' because the EEG buffer is empty."
                    )
                    return

                # OpenBCI GUI timestamps EEG in a different absolute clock than LSL markers.
                # Estimate the current offset and convert the marker to the EEG timebase.
                now_local = pylsl.local_clock()
                eeg_clock_offset = float(eeg_times[-1]) - now_local
                if self.marker_time_correction_valid:
                    marker_local_time = marker_lsl_time + self.marker_time_correction_s
                    marker_age_s = max(0.0, now_local - marker_local_time)
                    if marker_age_s > self.max_marker_age_s:
                        logger.warning(
                            f"[CHECK] Ignored late marker '{self.start_eval_marker}' "
                            f"with marker_age={marker_age_s:.3f}s. No decoding started."
                        )
                        return
                    marker_time = marker_local_time + eeg_clock_offset
                else:
                    # If LSL cannot provide marker clock correction, the marker
                    # timestamp may be in an incompatible clock. Use arrival time
                    # mapped to the newest stable EEG sample, so a valid trial is
                    # not discarded as "late" or just outside the buffer edge.
                    marker_age_s = 0.0
                    marker_time = max(
                        float(eeg_times[0]),
                        float(eeg_times[-1]) - self.fallback_marker_delay_s,
                    )
                    logger.warning(
                        f"[CHECK] Marker clock correction unavailable; using latest EEG time "
                        f"minus {self.fallback_marker_delay_s:.3f}s for '{self.start_eval_marker}'."
                    )

                buffer_tolerance_s = max(0.5, self.t_sleep_s * 2)
                if marker_time < eeg_times[0] and eeg_times[0] - marker_time <= buffer_tolerance_s:
                    marker_time = float(eeg_times[0])
                elif marker_time > eeg_times[-1] and marker_time - eeg_times[-1] <= buffer_tolerance_s:
                    marker_time = float(eeg_times[-1])

                if not (eeg_times[0] <= marker_time <= eeg_times[-1]):
                    logger.warning(
                        f"[CHECK] Ignored stale marker '{self.start_eval_marker}' "
                        f"with marker_age={marker_age_s:.3f}s because converted marker time "
                        f"{marker_time:.3f} is outside EEG buffer "
                        f"[{eeg_times[0]:.3f}, {eeg_times[-1]:.3f}]. No decoding started."
                    )
                    return

                logger.info(
                    f"[CHECK] Received marker '{self.start_eval_marker}' with marker_age={marker_age_s:.3f}s; "
                    f"converted to EEG time {marker_time:.3f}: decoding started."
                )
                self.is_decoding = True
                self.internal_decoding_start_time = time.time()

                self.start_eval_time = marker_time

    def check_udp_marker_should_start(self) -> bool:
        if self.marker_udp_socket is None:
            return False

        started = False
        while True:
            try:
                data, _ = self.marker_udp_socket.recvfrom(2048)
            except BlockingIOError:
                break
            except OSError as err:
                logger.warning(f"[CHECK] Could not read UDP marker socket: {err}")
                break
            payload = data.decode("utf-8", errors="replace").strip()
            arrival_time = time.time()
            parts = payload.split(":", 1)
            marker = parts[0]
            trial_id = None
            if len(parts) == 2:
                try:
                    trial_id = int(parts[1])
                except ValueError:
                    logger.warning(f"[CHECK] Ignored UDP marker with bad trial id: {payload}")
                    continue

            if marker == self.force_decision_marker:
                if trial_id is not None and trial_id != self.current_trial_id:
                    logger.info(
                        f"[CHECK] Ignored UDP force_decision for trial {trial_id}; "
                        f"current trial is {self.current_trial_id}."
                    )
                    continue
                if self.is_decoding:
                    self.force_decision_requested = True
                    logger.info(
                        f"[CHECK] Received UDP force_decision request for trial "
                        f"{self.current_trial_id}."
                    )
                else:
                    logger.info("[CHECK] Ignored UDP force_decision request because decoder is not decoding.")
                continue

            if marker != self.start_eval_marker:
                continue

            if self.is_decoding:
                logger.info(
                    f"[CHECK] Ignored UDP marker '{payload}' because trial "
                    f"{self.current_trial_id} is already decoding."
                )
                continue

            eeg_times = self.input_sw.unfold_buffer_t()
            if len(eeg_times) == 0:
                logger.warning(
                    f"[CHECK] Ignored UDP marker '{payload}' because the EEG buffer is empty."
                )
                continue

            marker_time = max(
                float(eeg_times[0]),
                float(eeg_times[-1]) - self.fallback_marker_delay_s,
            )
            marker_age_s = max(0.0, time.time() - arrival_time)
            logger.info(
                f"[CHECK] Received UDP marker '{payload}' with marker_age={marker_age_s:.3f}s; "
                f"using EEG time {marker_time:.3f}: decoding started for trial {trial_id}."
            )
            self.is_decoding = True
            self.current_trial_id = trial_id
            self.force_decision_requested = False
            self.internal_decoding_start_time = time.time()
            self.start_eval_time = marker_time
            started = True

        return started

    def _udp_marker_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            if self.marker_udp_socket is None:
                time.sleep(0.02)
                continue

            try:
                data, _ = self.marker_udp_socket.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError as err:
                if not stop_event.is_set():
                    logger.warning(f"[CHECK] UDP marker listener stopped/read failed: {err}")
                continue

            if self.marker_udp_queue is None:
                continue
            self.marker_udp_queue.put((
                data.decode("utf-8", errors="replace").strip(),
                time.time(),
            ))

    def _udp_marker_worker_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            if self.marker_udp_queue is None:
                time.sleep(0.02)
                continue

            try:
                first_packet = self.marker_udp_queue.get(timeout=0.02)
            except queue.Empty:
                continue

            packets = [first_packet]
            while True:
                try:
                    packets.append(self.marker_udp_queue.get_nowait())
                except queue.Empty:
                    break

            for payload, arrival_time in packets:
                self._handle_udp_marker_payload(payload, arrival_time=arrival_time)

    def _parse_udp_trial_packet(self, payload: str) -> tuple[str, int | None, float | None] | None:
        parts = payload.split(":")
        marker = parts[0]
        trial_id = None
        sent_time = None
        if len(parts) >= 2:
            try:
                trial_id = int(parts[1])
            except ValueError:
                logger.warning(f"[CHECK] Ignored UDP marker with bad trial id: {payload}")
                return None
        if len(parts) >= 3:
            try:
                sent_time = float(parts[2])
            except ValueError:
                logger.warning(f"[CHECK] Ignored UDP marker with bad timestamp: {payload}")
                return None

        return marker, trial_id, sent_time

    def _latest_udp_trial_packets(self, packets: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Drop queued stale trial packets so the decoder answers the current trial."""

        parsed_packets: list[tuple[str, int, str, float, float | None]] = []
        passthrough_packets: list[tuple[str, float]] = []
        for payload, arrival_time in packets:
            parsed = self._parse_udp_trial_packet(payload)
            if parsed is None:
                continue

            marker, trial_id, sent_time = parsed
            is_trial_marker = marker in {self.start_eval_marker, self.force_decision_marker}
            if is_trial_marker and trial_id is not None:
                parsed_packets.append((marker, trial_id, payload, arrival_time, sent_time))
            else:
                passthrough_packets.append((payload, arrival_time))

        if not parsed_packets:
            return passthrough_packets

        latest_trial_id = max(packet[1] for packet in parsed_packets)
        latest_packets = [
            packet for packet in parsed_packets if packet[1] == latest_trial_id
        ]
        marker_order = {
            self.start_eval_marker: 0,
            self.force_decision_marker: 1,
        }
        latest_packets.sort(key=lambda packet: marker_order.get(packet[0], 2))

        n_dropped = len(parsed_packets) - len(latest_packets)
        if n_dropped > 0:
            logger.info(
                f"[CHECK] Dropped {n_dropped} stale UDP trial packet(s); "
                f"processing latest trial {latest_trial_id}."
            )

        return passthrough_packets + [
            (payload, arrival_time)
            for _, _, payload, arrival_time, _ in latest_packets
        ]

    def _handle_udp_marker_payload(self, payload: str, arrival_time: float) -> None:
        parsed = self._parse_udp_trial_packet(payload)
        if parsed is None:
            return

        marker, trial_id, sent_time = parsed

        if not self.udp_decoding_enabled:
            logger.info(f"[CHECK] Ignored UDP marker '{payload}' because DECODE ONLINE is not active.")
            return

        now = time.time()
        packet_age_s = max(0.0, arrival_time - sent_time) if sent_time is not None else 0.0
        processing_age_s = max(0.0, now - sent_time) if sent_time is not None else 0.0
        processing_delay_s = max(0.0, processing_age_s - packet_age_s)
        is_stale_packet = (
            sent_time is not None
            and packet_age_s > self.max_udp_packet_age_s
        )
        if sent_time is not None and processing_delay_s > 0.25:
            logger.warning(
                f"[CHECK] UDP marker '{marker}' for trial {trial_id} was captured "
                f"after {packet_age_s:.3f}s but processed {processing_delay_s:.3f}s "
                "later; fast receiver kept the packet timestamp."
            )
        if is_stale_packet:
            logger.warning(
                f"[CHECK] Recovering late UDP marker '{marker}' for trial {trial_id}; "
                f"capture_age={packet_age_s:.3f}s. The old code would have dropped "
                "this trial and produced no UI output."
            )

        if marker == self.force_decision_marker:
            self._handle_udp_force_decision(trial_id, sent_time=sent_time)
            return

        if marker != self.start_eval_marker:
            return

        if self.is_decoding:
            if (
                    trial_id is None
                    or self.current_trial_id is None
                    or trial_id <= self.current_trial_id
            ):
                logger.info(
                    f"[CHECK] Ignored UDP marker '{payload}' because trial "
                    f"{self.current_trial_id} is already decoding."
                )
                return
            logger.info(
                f"[CHECK] Replacing stale decoder trial {self.current_trial_id} "
                f"with newer UDP trial {trial_id}."
            )

        self.is_decoding = True
        self.current_trial_id = trial_id
        self.force_decision_requested = False
        self.internal_decoding_start_time = now
        self.start_eval_time = 0.0
        logger.info(
            f"[CHECK] Trial {trial_id}: UDP start marker recorded without per-trial "
            "EEG flush, so marker handling stays real-time."
        )
        logger.info(
            f"[CHECK] Received UDP marker '{payload}' with capture_age={packet_age_s:.3f}s "
            f"and processing_age={processing_age_s:.3f}s; "
            f"decoder armed immediately for trial {trial_id}."
        )
        self._send_udp_decision(f"armed:{trial_id}")
        max_eval_time_s = self._max_eval_time_for_trial(trial_id)
        if processing_age_s >= max(0.2, max_eval_time_s - 0.25):
            logger.info(
                f"[CHECK] Trial {trial_id}: marker already covers the decision "
                "window; running rCCA immediately."
            )
            self._handle_udp_force_decision(trial_id, sent_time=sent_time)
        else:
            self._schedule_udp_auto_decision(
                trial_id,
                delay_s=max(0.1, max_eval_time_s - processing_age_s),
            )

    def _schedule_udp_auto_decision(
            self,
            trial_id: int | None,
            delay_s: float | None = None,
    ) -> None:
        if trial_id is None:
            return
        if self.udp_trial_timer is not None:
            self.udp_trial_timer.cancel()

        max_eval_time_s = self._max_eval_time_for_trial(trial_id)
        delay_s = max(0.1, float(max_eval_time_s if delay_s is None else delay_s))
        self.udp_trial_timer = threading.Timer(
            delay_s,
            self._auto_udp_force_decision,
            args=(trial_id,),
        )
        self.udp_trial_timer.daemon = True
        self.udp_trial_timer.start()
        logger.info(
            f"[CHECK] Trial {trial_id}: scheduled automatic rCCA decision "
            f"in {delay_s:.3f}s."
        )

    def _auto_udp_force_decision(self, trial_id: int) -> None:
        if trial_id in self.completed_udp_trial_ids:
            return
        if not self.is_decoding or self.current_trial_id != trial_id:
            logger.info(
                f"[CHECK] Trial {trial_id}: automatic rCCA decision skipped "
                f"(current trial is {self.current_trial_id}, decoding={self.is_decoding})."
            )
            return

        logger.info(f"[CHECK] Trial {trial_id}: automatic rCCA decision starting.")
        self._handle_udp_force_decision(trial_id, sent_time=None)

    def _handle_udp_force_decision(self, trial_id: int | None, sent_time: float | None = None) -> None:
        if trial_id is not None and trial_id in self.completed_udp_trial_ids:
            logger.info(
                f"[CHECK] Ignored duplicate force_decision for completed trial {trial_id}."
            )
            return

        max_eval_time_s = self._max_eval_time_for_trial(trial_id)

        if trial_id is not None and trial_id != self.current_trial_id:
            logger.info(
                f"[CHECK] Force decision for trial {trial_id} replaced stale "
                f"decoder trial {self.current_trial_id}."
            )
            self.current_trial_id = trial_id
            self.start_eval_time = 0.0
            self.internal_decoding_start_time = (
                sent_time - max_eval_time_s
                if sent_time is not None
                else time.time() - max_eval_time_s
            )

        if not self.is_decoding:
            logger.info(
                f"[CHECK] Force decision for trial {trial_id} armed decoder immediately."
            )
            self.is_decoding = True
            self.current_trial_id = trial_id
            self.start_eval_time = 0.0
            self.internal_decoding_start_time = (
                sent_time - max_eval_time_s
                if sent_time is not None
                else time.time() - max_eval_time_s
            )

        logger.info(f"[CHECK] Received UDP force_decision request for trial {self.current_trial_id}.")
        with self.processing_lock:
            self._update_eeg_filter_once()
            eeg_times = self.filterbank.ring_buffer.unfold_buffer_t()
            eeg_times = eeg_times[np.isfinite(eeg_times)]
            if len(eeg_times) == 0:
                logger.warning(
                    f"[CHECK] Trial {self.current_trial_id}: EEG buffer is empty; no rCCA output sent."
                )
                self.is_decoding = False
                self.current_trial_id = None
                return

            if self.start_eval_time <= 0:
                elapsed_s = max(0.0, time.time() - self.internal_decoding_start_time)
                requested_window_s = max(
                    elapsed_s,
                    max_eval_time_s + max(0.0, self.padding_size_s or 0.0),
                )
                requested_window_s = min(
                    requested_window_s,
                    max(0.0, self.buffer_size_s - 0.1),
                )
                self.start_eval_time = max(
                    float(eeg_times[0]),
                    float(eeg_times[-1]) - requested_window_s,
                )
            x = self._create_epoch()
            if x.shape[2] <= 0:
                logger.warning(
                    f"[CHECK] Trial {self.current_trial_id}: empty epoch at force decision; "
                    "no rCCA output sent."
                )
                self.is_decoding = False
                self.current_trial_id = None
                return

            xs = self._resample(x)
            if not self._force_classify(xs, reason="force_decision request"):
                logger.warning(
                    f"[CHECK] Trial {self.current_trial_id}: no rCCA score output available."
                )
                self.is_decoding = False
                self.current_trial_id = None

    def _update_eeg_filter_once(self) -> bool:
        if self.input_sw is None or self.filterbank is None:
            return False
        try:
            self.input_sw.update()
            self._filter()
        except Exception as err:
            logger.warning(f"[CHECK] Could not update EEG/filter for UDP decision: {err}")
            return False
        return True

    def _filter(self):
        if self.input_sw is None:
            logger.error("No input stream connected, cannot filter input.")

        if self.input_sw.n_new > 0:
            x = self.input_sw.unfold_buffer()[-self.input_sw.n_new:, self.selected_ch_idx]
            t = self.input_sw.unfold_buffer_t()[-self.input_sw.n_new:]
            self.filterbank.filter(x, t)
            self.input_sw.n_new = 0  # makes sure samples are filtered only once

    def _create_epoch(self) -> NDArray:
        x = self.filterbank.get_data()[:, :, 0]
        t = self.filterbank.ring_buffer.unfold_buffer_t()[-x.shape[0]:]

        if x.shape[0] != len(t):
            n = min(x.shape[0], len(t))
            logger.info(
                "[CHECK] Aligning filtered EEG/timestamp buffers: "
                f"x_samples={x.shape[0]}, t_samples={len(t)}, using {n}."
            )
            x = x[-n:, :]
            t = t[-n:]

        # Find marker onset timepoint
        idx = np.argmin(np.abs(t - self.start_eval_time))

        # Add padding interval to catch filtering artefacts
        if self.padding_size_s is not None and self.padding_size_s > 0:
            pad = int(self.input_sfreq * self.padding_size_s)
            idx = max(0, idx - pad)

        # Select trial relevant data
        x = x[idx:, :].T[None, :, :]  # (1, n_channels, n_samples)

        if np.isnan(x).sum() > 0:
            logger.error("NaNs found after epoching.")

        return x

    def _resample(self, x: NDArray) -> NDArray:

        if self.classifier_input_sfreq is not None:
            x = resample(
                x,
                num=int(x.shape[2] / self.input_sfreq * self.classifier_input_sfreq),
                axis=2,
            )

            # Remove padding interval to catch filtering artefacts
            if self.padding_size_s is not None and self.padding_size_s > 0:
                pad = int(self.classifier_input_sfreq * self.padding_size_s)
                x = x[:, :, pad:]

            if np.isnan(x).sum() > 0:
                logger.error("NaNs found after resampling")

        return x

    def _classify(self, x: NDArray):  # (n_trials, n_channels, n_samples) N.B. n_trials=1

        if x.shape[2] < int(self.t_sleep_s * self.classifier_input_sfreq):
            logger.debug(f"Classifying skipped as insufficient data: {x.shape[2]=}.")
            if time.time() - self.last_insufficient_log_time > 1.0:
                logger.info(
                    "[CHECK] Decoder has marker but not enough usable EEG yet: "
                    f"{x.shape[2]} samples after padding/resampling."
                )
                self.last_insufficient_log_time = time.time()
            y = -1
        else:
            y = self.classifier.predict(x)[0]   #predicts the selected key
            if y >= 0:
                logger.info(f"[CHECK] Decoder classified with prediction {y}.")

        # If y=-1 then the classifier is not yet sufficiently certain to emit the classification
        if y >= 0:
            self._push_decision(y, x)

    def _force_classify(self, x: NDArray, reason: str) -> bool:
        y = self._best_effort_class(x)
        if y < 0:
            logger.warning(
                f"[CHECK] Decoder {reason} reached but no rCCA score class could be computed."
            )
            return False

        logger.info(
            f"[CHECK] Decoder selected highest-score rCCA class {y} after {reason}."
        )
        self._push_decision(y, x)
        return True

    def _best_effort_class(self, x: NDArray) -> int:
        estimator = getattr(self.classifier, "estimator", self.classifier)
        if not hasattr(estimator, "decision_function"):
            return -1

        try:
            scores = np.asarray(estimator.decision_function(x)[0], dtype=float)
        except Exception as err:
            logger.warning(f"[CHECK] Could not compute rCCA class scores: {err}")
            return -1

        n_positions = self.n_positions if self.n_positions > 0 else len(scores)
        n_positions = min(n_positions, len(scores))
        if n_positions <= 0:
            return -1

        scores = scores[:n_positions]
        if not np.isfinite(scores).any():
            return -1

        return int(np.nanargmax(scores))

    def _push_decision(self, y: int, x: NDArray) -> None:
        decision_trial_id = self.current_trial_id
        if self.udp_trial_timer is not None:
            self.udp_trial_timer.cancel()
            self.udp_trial_timer = None
        if self.soft_decision_enabled:
            position = self._soft_decision_position(x)
            logger.info(f"[CHECK] Decoder pushing coordinate {position}.")
            self.output_sw.push_sample(
                [np.float32(position[0]), np.float32(position[1])]
            )
            self._send_udp_decision(
                f"coordinate:{self.current_trial_id}:{position[0]:.6f},{position[1]:.6f}"
            )
        else:
            logger.info(f"[CHECK] Decoder pushing class {y}.")
            self.output_sw.push_sample([np.int64(y)])
            self._send_udp_decision(f"class:{self.current_trial_id}:{int(y)}")
        if decision_trial_id is not None:
            self.completed_udp_trial_ids.add(decision_trial_id)
        self.is_decoding = False
        self.current_trial_id = None

    def _send_udp_decision(self, payload: str) -> None:
        if self.decoder_output_udp_socket is None or self.decoder_output_udp_port is None:
            return

        try:
            self.decoder_output_udp_socket.sendto(
                payload.encode("utf-8"),
                (self.decoder_output_udp_host, self.decoder_output_udp_port),
            )
        except OSError as err:
            logger.warning(
                f"[CHECK] Could not send UDP decoder output to "
                f"{self.decoder_output_udp_host}:{self.decoder_output_udp_port}: {err}"
            )
            return

        logger.info(
            f"[CHECK] Sent UDP decoder output '{payload}' to "
            f"{self.decoder_output_udp_host}:{self.decoder_output_udp_port}."
        )

    def _soft_decision_position(self, x: NDArray) -> NDArray:
        """Convert all rCCA class scores into a smoothed 2D position."""

        estimator = getattr(self.classifier, "estimator", self.classifier)
        scores = estimator.decision_function(x)[0]
        n_positions = self.n_positions if self.n_positions > 0 else len(scores)
        n_positions = min(n_positions, len(scores))

        scores = scores[:n_positions]
        positions = self._position_grid(n_positions)
        weights = self._softmax(scores)

        raw_position = weights @ positions
        confidence = self._weight_margin(weights)

        if self.last_soft_position is None:
            position = raw_position
        elif confidence < self.soft_decision_min_confidence:
            position = self.last_soft_position
            logger.debug(
                f"Keeping previous soft position because {confidence=} is below "
                f"{self.soft_decision_min_confidence=}."
            )
        else:
            alpha = np.clip(self.soft_decision_smoothing * confidence, 0.0, 1.0)
            position = (1.0 - alpha) * self.last_soft_position + alpha * raw_position

        self.last_soft_position = position
        logger.debug(
            f"Soft decision: {scores=}, {weights=}, {raw_position=}, "
            f"{confidence=}, final_position={position}."
        )

        return position

    def _position_grid(self, n_positions: int) -> NDArray:
        """Create normalized row/column tile coordinates in row-major order."""

        grid_width = int(np.ceil(np.sqrt(n_positions)))
        grid_height = int(np.ceil(n_positions / grid_width))

        positions = []
        for idx in range(n_positions):
            row = idx // grid_width
            col = idx % grid_width
            x_pos = 0.5 if grid_width == 1 else col / (grid_width - 1)
            y_pos = 0.5 if grid_height == 1 else row / (grid_height - 1)
            positions.append([x_pos, y_pos])

        return np.asarray(positions, dtype=float)

    def _softmax(self, scores: NDArray) -> NDArray:
        temperature = max(float(self.soft_decision_temperature), np.finfo(float).eps)
        scaled_scores = scores / temperature
        scaled_scores = scaled_scores - np.max(scaled_scores)
        weights = np.exp(scaled_scores)
        return weights / np.sum(weights)

    @staticmethod
    def _weight_margin(weights: NDArray) -> float:
        if weights.size < 2:
            return 1.0
        sorted_weights = np.sort(weights)
        return float(sorted_weights[-1] - sorted_weights[-2])

    def _run_loop(self, stop_event: threading.Event):

        logger.debug("Starting the run loop")
        if self.input_sw is None or self.output_sw is None or self.classifier is None:
            logger.error("Streams or decoding not initialized, call init_all first.")

        while not stop_event.is_set():
            t_start = pylsl.local_clock()
            self.update()
            t_end = pylsl.local_clock()

            # reduce sleep by processing time
            dt_sleep = self.t_sleep_s - (t_end - t_start)
            sleep_s(dt_sleep)


def online_decoder_factory(
        config_path: Path = Path("./configs/decoder_unity.toml"), preload: bool = True
):
    """Factory function to create an OnlineDecoder object from a config file."""
    cfg = toml.load(config_path)

    online_dec = OnlineDecoder(
        decoder_file=cfg["decoder"]["decoder_file"],
        decoder_meta_file=cfg["decoder"]["decoder_meta_file"],
        marker_stream_name=cfg["streams"]["marker_stream_name"],
        marker_udp_host=cfg["streams"].get("marker_udp_host", "127.0.0.1"),
        marker_udp_port=cfg["streams"].get("marker_udp_port", None),
        decoder_output_udp_host=cfg["streams"].get("decoder_output_udp_host", "127.0.0.1"),
        decoder_output_udp_port=cfg["streams"].get("decoder_output_udp_port", None),
        data_stream_name=cfg["streams"]["data_stream_name"],
        decoder_stream_name=cfg["streams"]["decoder_stream_name"],
        buffer_size_s=cfg["streams"]["buffer_size_s"],
        padding_size_s=cfg["streams"]["padding_size_s"],
        start_eval_marker=cfg["stimulus"]["trial_marker"],
        max_eval_time_s=cfg["online"]["max_eval_time_s"],
        first_trial_max_eval_time_s=cfg["online"].get("first_trial_max_eval_time_s", None),
        t_sleep_s=cfg["online"].get("sleep_s", 0.1),
        selected_channels=cfg["data"].get("selected_channels", None),
        n_positions=cfg["stimulus"].get("n_keys", 0),
        soft_decision_enabled=cfg["decoder"].get("soft_decision_enabled", False),
        soft_decision_smoothing=cfg["decoder"].get("soft_decision_smoothing", 0.5),
        soft_decision_min_confidence=cfg["decoder"].get("soft_decision_min_confidence", 0.0),
        soft_decision_temperature=cfg["decoder"].get("soft_decision_temperature", 1.0),
    )

    if preload:
        online_dec.load_model()

    return online_dec


def cli_run_decoder(
        conf_pth: Path = Path("./configs/decoder_unity.toml"), log_level: int = 30
):
    # if the CLI is run, we most likely also want a console output
    logger = get_logger("cvep_decoder", add_console_handler=True)
    logger.setLevel(log_level)

    logger.debug(f"Starting the decoder with {conf_pth=}")
    online_dec = online_decoder_factory(conf_pth)
    online_dec.init_all()
    thread, stop_event = online_dec.run()
    return thread, stop_event


if __name__ == "__main__":
    Fire(cli_run_decoder)
