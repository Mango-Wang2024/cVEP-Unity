import json
import os
import random
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import autocomplete
import google.generativeai as genai
import numpy as np
import psychopy
import pyttsx3
import toml
from dareplane_utils.logging.logger import get_logger
from fire import Fire
from psychopy import event, misc, monitors, visual
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop

from cvep_speller.utils.logging import logger

REALTIME_UDP_FIX_VERSION = "udp-speller-v7-main-thread-udp-polling"

# Windows does not allow / , : * ? " < > | ~ in file names (for the images)
KEY_MAPPING = {
    "slash": "/",
    "comma": ",",
    "colon": ":",
    "asterisk": "*",
    "question": "?",
    "quote": '"',
    "smaller": "<",
    "larger": ">",
    "bar": "|",
    "tilde": "~",
    "backslash": "\\",
    "backspace": "<-",
    "clear": "<<",
    "autocomplete": ">>",
}


class Speller(object):
    """
    An object to create a speller with keys and text fields. Keys can alternate their background images according to
    specifically setup stimulation sequences.

    Parameters
    ----------
    screen_resolution: tuple[int, int]
        The screen resolution in pixels, provided as (width, height).
    width_cm: float
        The width of the screen in cm to compute pixels per degree.
    distance_cm: float
        The distance of the screen to the user in cm to compute pixels per degree.
    refresh_rate: int
        The screen refresh rate in Hz.
    cfg: dict
        config object containing context and paradigm configuration info as loaded
        from `./configs/speller.toml`.
    screen_id: int (default: 0)
        The screen number where to present the keyboard when multiple screens are used.
    background_color: tuple[float, float, float] (default: (0., 0., 0.)
        The keyboard's background color specified as list of RGB values.
    marker_stream_name: str (default: "marker-stream")
        The name of the LSL stream to which markers of the keyboard are logged.
    quit_controls: list[str] (default: None)
        A list of keys that can be pressed to initiate quiting of the speller.
    full_screen: bool (default: True)
        Whether to present the speller in full screen mode.

    Attributes
    ----------
    keys: dict
        A dictionary of keys with a mapping of key name to a list of PsychoPy ImageStim.
    text_fields: dict
        A dictionary of text fields with a mapping of text field name to PsychoPy TextBox2.
    """

    def __init__(
        self,
        screen_resolution: tuple[int, int],
        width_cm: float,
        distance_cm: float,
        refresh_rate: int,
        cfg: dict,
        screen_id: int = 0,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        marker_stream_name: str = "marker-stream",
        quit_controls: list[str] = None,
        full_screen: bool = True,
    ) -> None:
        self.cfg = cfg
        self.screen_resolution = screen_resolution
        self.full_screen = full_screen
        self.width_cm = width_cm
        self.distance_cm = distance_cm
        self.refresh_rate = refresh_rate
        self.quit_controls = quit_controls
        self.keys: dict = {}
        self.keys_shift: dict = {}
        self.text_fields: dict = {}

        # Setup monitor
        self.monitor = monitors.Monitor(
            name="testMonitor", width=width_cm, distance=distance_cm
        )
        self.monitor.setSizePix(screen_resolution)

        # Setup window
        self.window = visual.Window(
            monitor=self.monitor,
            screen=screen_id,
            units="pix",
            size=screen_resolution,
            color=background_color,
            fullscr=full_screen,
            # infoMsg="",
        )
        self.window.setMouseVisible(False)

        self.marker_stream_name = marker_stream_name
        self.outlet = None
        self.decoder_marker_udp_host = cfg["streams"].get("marker_udp_host", "127.0.0.1")
        self.decoder_marker_udp_port = cfg["streams"].get("marker_udp_port", None)
        self.decoder_marker_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.decoder_output_udp_host = cfg["streams"].get("decoder_output_udp_host", "127.0.0.1")
        self.decoder_output_udp_port = cfg["streams"].get("decoder_output_udp_port", None)
        self.decoder_output_socket = None
        self.decoder_output_stop_event = None
        self.decoder_output_thread = None
        self.decoder_output_lock = threading.Lock()
        self.pending_decoder_arms: set[int] = set()
        self.pending_decoder_outputs: dict[int, str] = {}
        if self.decoder_output_udp_port is not None:
            self.decoder_output_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.decoder_output_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.decoder_output_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1_048_576)
            self.decoder_output_socket.bind((self.decoder_output_udp_host, self.decoder_output_udp_port))
            self.decoder_output_socket.setblocking(False)
            self.decoder_output_stop_event = threading.Event()
            logger.info(
                f"[CHECK] UDP decoder output main-thread listener ready on "
                f"{self.decoder_output_udp_host}:{self.decoder_output_udp_port}."
            )
            logger.info(
                f"[CHECK] Real-time UDP fix active: {REALTIME_UDP_FIX_VERSION}."
            )

        self.last_selected_key_idx: int | None = None
        self.last_selected_position: tuple[float, float] | None = None
        self.last_selected_label: str | None = None
        self.online_accuracy_rows: list[dict] = []
        self.key_map: dict[int, str] = {}
        self.highlights: dict = {}
        self.decoder_inlet = None
        self.decoding_event_seen = False
        self.current_trial_id: int | None = None
        self.accept_decoder_output = False
        self.skip_next_udp_trial_start = False

        # Set up variables for text to speech, autocompletion, and shifting keyboard layout
        self.sample_idx = 0  # used to iterate through sample symbols, iterated in handle_decoding_event
        # self.sample_symbols = ["shift","H","shift","e","l","l","o"," ","w","o","r","l","d",
        #                        " ", "shift", "I", "shift", "a", "m"]  # sample symbols for the speller to receive
        self.sample_symbols = []

        self.all_keys = self.set_all_keys(
            cfg
        )  # map all keys to their counterparts (A - > a, ! -> 1, etc.)
        self.case_flag = False  # True=upper, False=lower, start lower

        if self.cfg["speller"]["autocomplete"]["enabled"]:
            self.next_autocomplete = (
                ""  # holds the next autocompletion result to be displayed
            )
            self.autocomplete_engine = self.init_autocomplete_engine()

        if self.cfg["speller"]["text2speech"]["enabled"]:
            self.text2speech_engine = self.init_text2speech()
            self.text2speech_flag = False  # used to queue up text to speech

    def add_key(
        self,
        name: str,
        images: list,
        images_lower: list,
        size: tuple[int, int],
        pos: tuple[int, int],
    ) -> None:
        """
        Add a key to the speller.

        Parameters
        ----------
        name: str
            The name of the key.
        images: list
            The list of images associated to the key. Note, index -1 fused for presenting feedback, and index -2 is
            used for cueing.
        images_lower: list
            The list of images associated to the key when the shift key is pressed. If empty, the images list is used.
            The same indices apply as for the images list.
        size: tuple[int, int]
            The size of the key in pixels provided as (width, height).
        pos: tuple[int, int]
            The position of the key in pixels provided as (x, y).
        """
        assert name not in self.keys, "Adding a key with a name that already exists!"
        self.keys[name] = []
        self.keys_shift[name] = []
        for image in images:
            self.keys[name].append(
                visual.ImageStim(
                    win=self.window,
                    name=name,
                    image=image,
                    units="pix",
                    pos=pos,
                    size=size,
                    autoLog=False,
                )
            )
        if len(images_lower) == 0:
            self.keys_shift[name] = self.keys[name]
        else:
            self.keys_shift[name] = []
            for image in images_lower:
                self.keys_shift[name].append(
                    visual.ImageStim(
                        win=self.window,
                        name=name,
                        image=image,
                        units="pix",
                        pos=pos,
                        size=size,
                        autoLog=False,
                    )
                )

        # Set autoDraw to True for first default key to keep app visible; check case_flag to determine which set of
        # keys to display at the start
        if self.case_flag:
            self.keys[name][0].setAutoDraw(True)
        else:
            self.keys_shift[name][0].setAutoDraw(True)

    def add_text_field(
        self,
        name: str,
        text: str,
        size: tuple[int, int],
        pos: tuple[int, int],
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        text_color: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        alignment: str = "left",
        bold: bool = False,
    ) -> None:
        """
        Add a text field to the speller.

        Parameters
        ----------
        name: str
            The name of the text field.
        text: str
            The text to display on the text field.
        size: tuple[int, int]
            The size of the text field in pixels provided as (width, height).
        pos: tuple[int, int]
            The position of the text field in pixels provided as (x, y).
        background_color: tuple[float, float, float] (default: (0., 0., 0.))
            The background color of the text field  specified as list of RGB values.
        text_color: tuple[float, float, float] (default: (-1., -1., -1.))
            The text color of the text field  specified as list of RGB values.
        alignment: str (default: "left")
            The alignment of the text in the text field.
        bold: bool (default: False)
            Whether the text is boldface.
        """
        assert name not in self.text_fields, (
            "Adding a text field with a name that already exists!"
        )
        self.text_fields[name] = visual.TextBox2(
            win=self.window,
            name=name,
            text=text,
            units="pix",
            pos=pos,
            size=size,
            letterHeight=0.8 * size[1],
            fillColor=background_color,
            color=text_color,
            alignment=alignment,
            autoDraw=True,
            autoLog=False,
            bold=bold,
        )

    def connect_to_decoder_lsl_stream(self) -> None:
        name = self.cfg["streams"]["decoder_stream_name"]
        logger.info(f'Connecting to decoder stream "{name}".')
        streams = []
        while len(streams) == 0:
            streams = resolve_byprop("name", name, timeout=1)
            if len(streams) == 0:
                logger.info(f'Waiting for decoder stream "{name}".')
        if len(streams) > 1:
            logger.warning(f'Selecting first decoder stream named "{name}".')
        self.decoder_inlet = StreamInlet(streams[0])
        logger.info(f'[CHECK] Decoder stream "{name}" connected.')

    def create_marker_lsl_stream(self) -> None:
        if self.outlet is not None:
            return
        info = StreamInfo(
            name=self.marker_stream_name,
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id=self.marker_stream_name,
        )
        self.outlet = StreamOutlet(info)
        logger.info(f'[CHECK] Marker stream "{self.marker_stream_name}" created.')

    def get_pixels_per_degree(
        self,
    ) -> float:
        """
        Get the pixels per degree of the screen.

        Returns
        -------
        ppd: float
            Pixels per degree of the screen.
        """
        return misc.deg2pix(degrees=1.0, monitor=self.monitor)

    def get_text_field(
        self,
        name: str,
    ) -> str:
        """
        Get the text of the text field.

        Parameters
        ----------
        name: str
            The name of the text field.

        Returns
        -------
        text: str
            The text of the text field.
        """
        assert name in self.text_fields, (
            "Getting text of a text field with a name that does not exists!"
        )
        return self.text_fields[name].text

    def set_text_field(
        self,
        name: str,
        text: str,
    ) -> None:
        """
        Set the text of a text field.

        Parameters
        ----------
        name: str
            The name of the text field.
        text: str
            The text of the text field.
        """
        assert name in self.text_fields, (
            "Setting text of a text field with a name that does not exists!"
        )
        self.text_fields[name].setText(text)
        self.window.flip()

    def log(
        self,
        marker: str,
        on_flip: bool = False,
    ) -> None:
        """
        Log a marker to the marker stream.

        Parameters
        ----------
        marker: str
            The marker to log.
        on_flip: bool (default: False)
            Whether to log on the next frame flip.
        """
        if self.outlet is None:
            raise RuntimeError("Marker stream has not been created yet.")
        if on_flip:
            self.window.callOnFlip(self.outlet.push_sample, [marker])
        else:
            self.send_decoder_marker(marker)
            self.outlet.push_sample([marker])

    def send_decoder_marker(self, marker: str) -> None:
        if self.decoder_marker_udp_port is None:
            return
        if marker != self.cfg["speller"]["markers"]["trial_start"]:
            return
        if self.skip_next_udp_trial_start:
            self.skip_next_udp_trial_start = False
            logger.info(
                f'[CHECK] Skipped duplicate UDP marker "{marker}" because '
                f"trial {self.current_trial_id} was already armed."
            )
            return
        if self.current_trial_id is None:
            logger.warning(
                f'[CHECK] Not sending UDP marker "{marker}" because no online trial id is active.'
            )
            return
        payload = f"{marker}:{self.current_trial_id}:{time.time():.6f}"
        try:
            self.decoder_marker_socket.sendto(
                payload.encode("utf-8"),
                (self.decoder_marker_udp_host, int(self.decoder_marker_udp_port)),
            )
        except OSError as err:
            logger.error(
                f'[CHECK] Failed to send UDP marker "{payload}" to '
                f"{self.decoder_marker_udp_host}:{self.decoder_marker_udp_port}: {err}."
            )
            raise
        logger.info(
            f'[CHECK] Sent UDP marker "{payload}" to '
            f"{self.decoder_marker_udp_host}:{self.decoder_marker_udp_port}."
        )

    def request_decoder_force_decision(self) -> None:
        if self.decoder_marker_udp_port is None:
            return
        if self.current_trial_id is None:
            logger.warning("[CHECK] Not sending force_decision because no online trial id is active.")
            return

        marker = f"force_decision:{self.current_trial_id}:{time.time():.6f}"
        try:
            self.decoder_marker_socket.sendto(
                marker.encode("utf-8"),
                (self.decoder_marker_udp_host, int(self.decoder_marker_udp_port)),
            )
        except OSError as err:
            logger.error(
                f'[CHECK] Failed to send UDP marker "{marker}" to '
                f"{self.decoder_marker_udp_host}:{self.decoder_marker_udp_port}: {err}."
            )
            raise
        logger.info(
            f'[CHECK] Sent UDP marker "{marker}" to '
            f"{self.decoder_marker_udp_host}:{self.decoder_marker_udp_port}."
        )

    def run(
        self,
        sequences: dict,
        duration: float = None,
        start_marker: str = None,
        stop_marker: str = None,
        start_marker_on_flip: bool = True,
        check_decoder_output: bool = True,
    ) -> None:
        """
        Run a stimulation phase of the speller, which makes the keys flash according to specific sequences.

        Parameters
        ----------
        sequences: dict
            A dictionary containing the stimulus sequences per key.
        duration: float (default: None)
            The duration of the stimulation in seconds. If None, the duration of the first key in the dictionary is
            used.
        start_marker: str (default: None)
            The marker to log when stimulation starts. If None, no marker is logged.
        stop_marker: str (default: None)
            The marker to log when stimulation stops. If None, no marker is logged.
        start_marker_on_flip: bool (default: True)
            Whether to send the start marker on the next frame flip.
        """

        # Set number of frames
        if duration is None:
            n_frames = len(sequences[list(sequences.keys())[0]])
        else:
            n_frames = int(duration * self.refresh_rate)

        # Set autoDraw to False for full control, check case_flag to determine which set of keys to stop drawing
        if self.case_flag:
            for key in self.keys.values():
                key[0].setAutoDraw(False)
        else:
            for key in self.keys_shift.values():
                key[0].setAutoDraw(False)

        # Send start marker
        if start_marker is not None:
            self.log(start_marker, on_flip=start_marker_on_flip)
            logger.info(
                f'[CHECK] Sent marker "{start_marker}" '
                f"(on_flip={start_marker_on_flip})."
            )

        # Loop frame flips
        for i in range(n_frames):
            stime = time.time()

            # Check quiting
            if i % 60 == 0:
                if len(event.getKeys(keyList=self.quit_controls)) > 0:
                    self.quit()
                    break

            # Check selection marker
            if (
                    check_decoder_output
                    and (
                        self.decoder_inlet is not None
                        or self.decoder_output_socket is not None
                    )
            ):
                if self.has_decoding_event():
                    self.handle_decoding_event()
                    break

            # Present keys with color depending on code state and case_flag
            if self.case_flag:
                for name, code in sequences.items():
                    self.keys[name][code[i % len(code)]].draw()
            else:
                for name, code in sequences.items():
                    self.keys_shift[name][code[i % len(code)]].draw()

            # Check if frame flip can happen within a frame
            etime = time.time() - stime
            if etime >= 1 / self.refresh_rate:
                logger.warn(f"Frame flip took too long ({etime:.6f}), dropping frames!")

            self.window.flip()
        else:
            logger.debug(f"All {n_frames=} shown.")

        # Send stop marker
        if stop_marker is not None:
            self.log(stop_marker)

        # Set autoDraw to True to keep speller visible after checking case_flag
        if self.case_flag:
            for key in self.keys.values():
                key[0].setAutoDraw(True)
        else:
            for key in self.keys_shift.values():
                key[0].setAutoDraw(True)

    def quit(
        self,
    ) -> None:
        """
        Quit the speller, close the window.
        """
        for key in self.keys.values():
            key[0].setAutoDraw(True)

        if self.window is not None:
            self.window.flip()
            self.window.setMouseVisible(True)
            self.window.close()

    def has_decoding_event(self) -> bool:
        """
        Check if the LSL stream contained a `speller_select <key_idx>` marker.
        """
        if self.decoding_event_seen or not self.accept_decoder_output:
            return False

        if self.decoder_output_socket is not None and self.has_udp_decoding_event():
            return True

        if self.decoder_inlet is None:
            return False

        latest_sample = None
        for _ in range(32):
            sample, _ = self.decoder_inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            latest_sample = sample

        if latest_sample is None:
            return False

        prediction = np.asarray(latest_sample)
        logger.debug(f"Received: prediction={prediction}")

        if prediction.ndim == 1 and prediction.shape[0] >= 2:
            x_pos, y_pos = prediction[:2]
            if x_pos >= 0 and y_pos >= 0:
                self.last_selected_key_idx = None
                self.last_selected_position = (float(x_pos), float(y_pos))
                self.decoding_event_seen = True
                logger.info(
                    f"[CHECK] Trial {self.current_trial_id}: speller received "
                    f"LSL fallback coordinate ({x_pos:.3f}, {y_pos:.3f})."
                )
                logger.debug(f"Soft position: {self.last_selected_position}")
                return True

        selections = prediction.flatten()
        selections = selections[selections >= 0]

        if len(selections) > 0:
            self.last_selected_key_idx = int(selections[-1])
            self.decoding_event_seen = True
            logger.info(
                f"[CHECK] Trial {self.current_trial_id}: speller received "
                f"LSL fallback class {self.last_selected_key_idx}."
            )
            logger.debug(f"Selection: {self.last_selected_key_idx}, {selections=}")
            return True

        return False

    def _decoder_output_loop(self) -> None:
        while (
                self.decoder_output_socket is not None
                and self.decoder_output_stop_event is not None
                and not self.decoder_output_stop_event.is_set()
        ):
            try:
                data, _ = self.decoder_output_socket.recvfrom(2048)
            except (BlockingIOError, socket.timeout):
                time.sleep(0.001)
                continue
            except OSError as err:
                if not self.decoder_output_stop_event.is_set():
                    logger.warning(f"[CHECK] UDP decoder output listener stopped/read failed: {err}")
                break

            payload = data.decode("utf-8", errors="replace").strip()
            self._store_udp_decoder_payload(payload)

    def poll_udp_decoder_outputs(self, max_packets: int = 128) -> int:
        if self.decoder_output_socket is None:
            return 0

        n_packets = 0
        for _ in range(max_packets):
            try:
                data, _ = self.decoder_output_socket.recvfrom(2048)
            except BlockingIOError:
                break
            except OSError as err:
                logger.warning(f"[CHECK] UDP decoder output polling failed: {err}")
                break

            payload = data.decode("utf-8", errors="replace").strip()
            self._store_udp_decoder_payload(payload)
            n_packets += 1

        return n_packets

    def _store_udp_decoder_payload(self, payload: str) -> None:
        parts = payload.split(":")
        if len(parts) >= 2 and parts[0] == "armed":
            try:
                trial_id = int(parts[1])
            except ValueError:
                logger.warning(f"[CHECK] Ignored malformed decoder arm response: {payload}")
                return
            with self.decoder_output_lock:
                self.pending_decoder_arms.add(trial_id)
            if trial_id == self.current_trial_id:
                logger.info(f"[CHECK] Trial {trial_id}: stored UDP decoder arm response.")
            return

        if len(parts) < 3 or parts[0] not in {"class", "coordinate"}:
            logger.warning(f"[CHECK] Ignored malformed UDP decoder output: {payload}")
            return

        try:
            trial_id = int(parts[1])
        except ValueError:
            logger.warning(f"[CHECK] Ignored UDP decoder output with bad trial id: {payload}")
            return

        with self.decoder_output_lock:
            self.pending_decoder_outputs[trial_id] = payload
        if trial_id == self.current_trial_id:
            logger.info(f"[CHECK] Trial {trial_id}: stored UDP decoder output {payload}.")
        else:
            logger.info(
                f"[CHECK] Stored UDP decoder output {payload}; current trial is "
                f"{self.current_trial_id}."
            )

    def has_udp_decoding_event(self) -> bool:
        if self.decoder_output_socket is None:
            return False

        self.poll_udp_decoder_outputs()

        with self.decoder_output_lock:
            accepted_payload = self.pending_decoder_outputs.pop(
                self.current_trial_id,
                None,
            )

        if accepted_payload is None:
            return False

        parts = accepted_payload.split(":", 2)
        kind = parts[0]
        value = parts[2]

        if kind == "class":
            try:
                self.last_selected_key_idx = int(value)
            except ValueError:
                logger.warning(f"[CHECK] Ignored malformed UDP decoder class: {accepted_payload}")
                return False
            self.last_selected_position = None
            self.decoding_event_seen = True
            logger.info(
                f"[CHECK] Trial {self.current_trial_id}: speller received UDP class "
                f"{self.last_selected_key_idx}."
            )
            return True

        if kind == "coordinate":
            try:
                x_text, y_text = value.split(",", 1)
                x_pos = float(x_text)
                y_pos = float(y_text)
            except ValueError:
                logger.warning(f"[CHECK] Ignored malformed UDP decoder coordinate: {accepted_payload}")
                return False
            self.last_selected_key_idx = None
            self.last_selected_position = (x_pos, y_pos)
            self.decoding_event_seen = True
            logger.info(
                f"[CHECK] Trial {self.current_trial_id}: speller received UDP coordinate "
                f"({x_pos:.3f}, {y_pos:.3f})."
            )
            return True

        return False

    def wait_for_decoder_arm(self, duration: float, retry_marker: str | None = None) -> bool:
        if duration <= 0 or self.decoder_output_socket is None:
            return True

        deadline = time.time() + duration
        next_retry_time = time.time() + 1.0
        while time.time() < deadline:
            self.poll_udp_decoder_outputs()
            with self.decoder_output_lock:
                if self.current_trial_id in self.pending_decoder_arms:
                    self.pending_decoder_arms.remove(self.current_trial_id)
                    logger.info(f"[CHECK] Trial {self.current_trial_id}: decoder armed before flashing.")
                    return True
                if self.current_trial_id in self.pending_decoder_outputs:
                    logger.info(
                        f"[CHECK] Trial {self.current_trial_id}: decoder output arrived "
                        "before arm wait ended; treating decoder as ready."
                    )
                    return True

            if retry_marker is not None and time.time() >= next_retry_time:
                logger.info(
                    f"[CHECK] Trial {self.current_trial_id}: retrying decoder arm marker."
                )
                self.send_decoder_marker(retry_marker)
                next_retry_time = time.time() + 1.0
            self.window.flip()

        logger.error(
            f"[CHECK] Trial {self.current_trial_id}: decoder did not arm within "
            f"{duration:.1f}s; skipping useful flashing would be safer."
        )
        return False

    def drain_decoder_stream(self) -> int:
        """Clear old decoder samples so they cannot be used for the next trial."""

        if self.decoder_inlet is None:
            return self.drain_udp_decoder_outputs()

        n_drained = 0
        for _ in range(128):
            sample, _ = self.decoder_inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            n_drained += 1

        return n_drained + self.drain_udp_decoder_outputs()

    def drain_udp_decoder_outputs(self) -> int:
        self.poll_udp_decoder_outputs()

        with self.decoder_output_lock:
            stale_output_trials = [
                trial_id
                for trial_id in self.pending_decoder_outputs
                if trial_id != self.current_trial_id
            ]
            stale_arm_trials = [
                trial_id
                for trial_id in self.pending_decoder_arms
                if trial_id != self.current_trial_id
            ]
            for trial_id in stale_output_trials:
                self.pending_decoder_outputs.pop(trial_id, None)
            for trial_id in stale_arm_trials:
                self.pending_decoder_arms.discard(trial_id)

        return len(stale_output_trials) + len(stale_arm_trials)

    def wait_for_decoding_event(self, duration: float, retry_force_decision: bool = False) -> bool:
        """Wait a little longer for decoder output after stimulation stops."""

        if duration <= 0:
            return False

        if self.decoder_inlet is None and self.decoder_output_socket is None:
            return False

        deadline = time.time() + duration
        while time.time() < deadline:
            if len(event.getKeys(keyList=self.quit_controls)) > 0:
                self.quit()
                return False

            if self.has_decoding_event():
                self.handle_decoding_event()
                return True

            self.window.flip()

        if self.has_decoding_event():
            logger.info(
                f"[CHECK] Trial {self.current_trial_id}: decoder output arrived "
                "at the timeout edge; accepting it before marking no-output."
            )
            self.handle_decoding_event()
            return True

        return False

    def get_online_accuracy_mode_label(self) -> str:
        plot_cfg = self.cfg.get("online_accuracy", {})
        configured_label = str(plot_cfg.get("mode_label", "")).strip()
        if configured_label:
            return configured_label

        decoder_config_file = plot_cfg.get(
            "decoder_config_file",
            "/Users/wang/dp-cvep-1/cvep_speller_env/dp-cvep-decoder/configs/decoder_unity.toml",
        )
        try:
            decoder_cfg = toml.load(decoder_config_file)
            mode = decoder_cfg["decoder"].get("training_mode", "unknown").lower()
        except Exception as err:
            logger.warning(f"[CHECK] Could not read decoder mode for accuracy plot: {err}")
            mode = "unknown"

        mode_labels = {
            "zero": "Zero-training",
            "calibration": "Calibration",
        }
        return mode_labels.get(mode, mode.title())

    def record_online_accuracy_trial(self, trial_id: int, true_target: str | None) -> None:
        predicted_target = self.last_selected_label if self.decoding_event_seen else None
        is_correct = (
            true_target is not None
            and predicted_target is not None
            and str(true_target) == str(predicted_target)
        )
        self.online_accuracy_rows.append(
            {
                "trial": trial_id,
                "mode": self.get_online_accuracy_mode_label(),
                "true_target": true_target if true_target is not None else "",
                "predicted_target": predicted_target if predicted_target is not None else "",
                "correct": int(is_correct),
            }
        )
        logger.info(
            f"[CHECK] Accuracy trial {trial_id}: true={true_target}, "
            f"predicted={predicted_target}, correct={is_correct}."
        )

    def save_online_accuracy_plot(self) -> None:
        plot_cfg = self.cfg.get("online_accuracy", {})
        if not plot_cfg.get("enabled", False):
            return
        if len(self.online_accuracy_rows) == 0:
            logger.warning("[CHECK] No online accuracy rows available; no accuracy plot saved.")
            return

        output_dir = Path(
            plot_cfg.get(
                "output_dir",
                "/Users/wang/dp-cvep-1/cvep_speller_env/data/online_accuracy",
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        mode_label = self.get_online_accuracy_mode_label()
        mode_slug = mode_label.lower().replace("-", "_").replace(" ", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        png_path = output_dir / f"online_accuracy_{mode_slug}_{timestamp}.png"

        correct = sum(int(row["correct"]) for row in self.online_accuracy_rows)
        n_trials = len(self.online_accuracy_rows)
        accuracy = correct / n_trials if n_trials > 0 else 0.0

        true_targets = [str(row["true_target"]) for row in self.online_accuracy_rows]
        predicted_targets = [
            str(row["predicted_target"]) if row["predicted_target"] != "" else "No output"
            for row in self.online_accuracy_rows
        ]
        target_labels = list(dict.fromkeys(
            [label for label in true_targets + predicted_targets if label != ""]
        ))
        target_labels = sorted(
            target_labels,
            key=lambda label: (
                label == "No output",
                0 if label.isdigit() else 1,
                int(label) if label.isdigit() else label,
            ),
        )
        label_to_y = {label: idx for idx, label in enumerate(target_labels)}

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "font.size": 10,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.dpi": 160,
                "savefig.dpi": 300,
            }
        )

        fig, ax_trial = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        trials = [int(row["trial"]) for row in self.online_accuracy_rows]
        true_y = [label_to_y.get(label, np.nan) for label in true_targets]
        pred_y = [label_to_y.get(label, np.nan) for label in predicted_targets]
        correct_flags = [bool(int(row["correct"])) for row in self.online_accuracy_rows]

        for trial, is_correct in zip(trials, correct_flags):
            ax_trial.axvspan(
                trial - 0.45,
                trial + 0.45,
                color=("#E5F4EA" if is_correct else "#FBE6E6"),
                alpha=0.75,
                linewidth=0,
            )
        ax_trial.plot(trials, true_y, "o-", color="#222222", label="True target", markersize=4)
        ax_trial.plot(trials, pred_y, "x--", color="#D65F00", label="Predicted target", markersize=5)
        ax_trial.set_yticks(range(len(target_labels)))
        ax_trial.set_yticklabels(target_labels)
        ax_trial.set_xlabel("Trial")
        ax_trial.set_ylabel("Target")
        ax_trial.set_title(
            f"Online cVEP Speller Accuracy - {mode_label}\n"
            f"Trial-by-Trial Prediction ({correct}/{n_trials}, {accuracy * 100:.1f}% correct)",
            fontweight="bold",
        )
        ax_trial.set_xlim(0.5, max(trials) + 0.5)
        ax_trial.legend(frameon=False, loc="upper right", fontsize=8)
        ax_trial.grid(axis="y", color="#DDDDDD", linewidth=0.6)

        fig.savefig(png_path, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"[CHECK] Saved online accuracy PNG: {png_path}")

        if plot_cfg.get("open_after_save", True):
            try:
                subprocess.Popen(["open", str(png_path)])
                logger.info(f"[CHECK] Opened online accuracy PNG: {png_path}")
            except Exception as err:
                logger.warning(f"[CHECK] Could not open online accuracy PNG automatically: {err}")

    def handle_decoding_event(self) -> None:
        # Decoding
        logger.info("Waiting for decoding")

        if self.last_selected_key_idx is None and self.last_selected_position is not None:
            x_pos, y_pos = self.last_selected_position
            position_text = f"({x_pos:.3f}, {y_pos:.3f})"
            self.last_selected_label = position_text
            self.set_text_field(name="text", text=position_text)
            logger.info(f"[CHECK] Speller showing coordinate {position_text}.")
            logger.debug(f"Feedback: soft_position={position_text}")
            self.run(
                sequences=self.highlights,
                duration=self.cfg["speller"]["timing"]["feedback_s"],
                start_marker=(
                    f"{self.cfg['speller']['markers']['feedback_start']};"
                    f"x={x_pos:.6f};y={y_pos:.6f}"
                ),
                stop_marker=self.cfg["speller"]["markers"]["feedback_stop"],
                check_decoder_output=False,
            )
            return

        # with fewer keys, it is possible for random value to be out of bounds so take idx % number of keys
        prediction = self.last_selected_key_idx % len(self.key_map)

        # additionally, the key map only includes capitalized versions of the keys, so shift the prediction if necessary
        if self.case_flag:
            prediction_key = self.key_map[
                prediction
            ]  # self.key_map[0] = tilde, for example
        else:
            prediction_key = self.all_keys[self.key_map[prediction]]
        self.last_selected_label = str(self.key_map[prediction])

        logger.debug(
            f"Decoding: prediction={prediction} prediction_key={prediction_key}"
        )

        # Spelling
        text = self.get_text_field("text")

        # update the autocomplete text field with the current text, update variable to hold the next autocompletion
        # result in case autocomplete key is pressed
        if self.cfg["speller"]["autocomplete"]["enabled"]:
            self.set_text_field(name="autocomplete_text", text=self.next_autocomplete)
            autocompleted_text = self.next_autocomplete

        # if there is an available list of sample symbols, use them, otherwise use the random prediction
        if self.sample_idx < len(self.sample_symbols):
            symbol = self.sample_symbols[self.sample_idx]
            if (
                symbol in KEY_MAPPING.values()
            ):  # update the prediction key to the recognized key name if the symbol is
                # a special character (i.e. / -> slash)
                prediction_key = list(KEY_MAPPING.keys())[
                    list(KEY_MAPPING.values()).index(symbol)
                ]
            elif symbol == " ":
                prediction_key = "space"
            else:
                prediction_key = symbol
            self.sample_idx += 1  # increment the sample index to move to the next symbol for the next decode event
        else:
            symbol = prediction_key

        keys_from_cfg = self.cfg["speller"]["keys"]["keys_upper"]
        grid_width = max(len(row) for row in keys_from_cfg)
        grid_y = prediction // grid_width
        grid_x = prediction % grid_width
        coordinate_text = f"({grid_x}, {grid_y})"

        # if the symbol is a key in KEY_MAPPING, change it to a symbol to be added to the text field
        # (i.e. "slash" -> "/")
        if symbol in KEY_MAPPING:
            symbol = KEY_MAPPING[symbol]

        # Handle special keys
        if symbol == self.cfg["speller"]["key_space"]:
            # Add a whitespace
            text = text + " "
        elif symbol == self.cfg["speller"]["key_clear"]:
            # Clear the full sentence
            text = ""
        elif symbol == self.cfg["speller"]["key_backspace"]:
            # Perform a backspace
            text = text[:-1]
        elif (
            self.cfg["speller"]["autocomplete"]["enabled"]
            and symbol == self.cfg["speller"]["key_autocomplete"]
        ):
            # If autocomplete enabled, update the text variable with the autocompleted text to be added to the text
            # field, if not enabled, treat symbol as text
            text = autocompleted_text
        elif symbol == self.cfg["speller"]["key_shift"]:
            # Update the shift flag to change the case of the keyboard for the next iteration
            self.case_flag = not self.case_flag
        elif symbol == self.cfg["speller"]["key_text2speech"]:
            # Enable text2speech flag to be spoken after feedback
            self.text2speech_flag = True
        else:
            text = coordinate_text

        # update the text field with the new text
        self.set_text_field(name="text", text=text)
        logger.debug(f"Feedback: symbol={symbol} text={text}")

        # if the updated text is not empty, start the autocomplete process with the updated text (if enabled)
        if len(text) >= 1 and self.cfg["speller"]["autocomplete"]["enabled"]:
            self.start_autocomplete()
        else:
            self.next_autocomplete = text

        # Feedback
        logger.info(f"Presenting feedback {prediction_key} ({prediction})")
        # if the prediction is in the second half of the key map, find its equivalent in the first half
        if not self.case_flag:
            prediction_key = self.all_keys[
                prediction_key
            ]  # set a -> A, etc. for highlights
        self.highlights[prediction_key] = [-1]
        self.run(
            sequences=self.highlights,
            duration=self.cfg["speller"]["timing"]["feedback_s"],
            start_marker=f"{self.cfg['speller']['markers']['feedback_start']};label={prediction};key={prediction_key}",
            stop_marker=self.cfg["speller"]["markers"]["feedback_stop"],
            check_decoder_output=False,
        )

        if self.cfg["speller"]["text2speech"]["enabled"]:
            # if text2speech flag is true and feedback is complete, speak the text
            if self.text2speech_flag:
                self.speak_text(text)
                self.text2speech_flag = (
                    False  # reset the text2speech flag for next decode event
                )

        # remove the highlight from the selected key
        self.highlights[prediction_key] = [0]

    def init_highlights_with_zero(self) -> None:
        # Setup highlights
        self.highlights = dict()
        keys_from_cfg = self.cfg["speller"]["keys"]["keys_upper"]

        for row in keys_from_cfg:
            for key in row:
                self.highlights[key] = [0]
        if self.cfg["speller"]["stt"]["enabled"]:
            self.highlights["stt"] = [0]

    def set_all_keys(self, cfg: dict) -> dict:
        """
        map all keys to their counterparts (A - > a, ! -> 1, etc.)
        """
        all_keys = {}
        for y in range(len(cfg["speller"]["keys"]["keys_upper"])):
            for x in range(len(cfg["speller"]["keys"]["keys_upper"][y])):
                all_keys[cfg["speller"]["keys"]["keys_upper"][y][x]] = cfg["speller"][
                    "keys"
                ]["keys_lower"][y][x]
                all_keys[cfg["speller"]["keys"]["keys_lower"][y][x]] = cfg["speller"][
                    "keys"
                ]["keys_upper"][y][x]
        return all_keys

    def init_text2speech(self) -> pyttsx3.init:
        """
        # return a pyttsx3 engine based on user's operating system, with the specified settings from config
        """
        engine = pyttsx3.init()
        voice_idx = self.cfg["speller"]["text2speech"][
            "voice_idx"
        ]  # 0 male, 1 female, can install more in system
        engine.setProperty("voice", engine.getProperty("voices")[voice_idx].id)
        engine.setProperty(
            "rate", self.cfg["speller"]["text2speech"]["rate"]
        )  # integer value for words/minute
        engine.setProperty(
            "volume", self.cfg["speller"]["text2speech"]["volume"]
        )  # float value from 0 to 1
        return engine

    def speak_text(self, text: str) -> None:
        """
        use the initialized text2speech engine to speak the text
        """
        try:
            self.text2speech_engine.say(text)
            self.text2speech_engine.runAndWait()
            self.text2speech_engine.stop()
        except Exception as e:
            print(f"text2speech Error: {e}")

    def init_autocomplete_engine(self) -> genai.GenerativeModel:
        """
        return a generative AI model based on the specified settings from config
        """
        # first, check if autocomplete is enabled in the config
        if self.cfg["speller"]["autocomplete"]["enabled"]:
            """
            models: list[str] - list of models to choose from:
            "gemini-1.5-pro": larger model with more parameters, better performance but slower (1.5s per request), 2 
                Requests per Minute limit
            "gemini-1.5-flash-8b", "gemini-1.5-flash": smaller models with less parameters, faster (~0.5-0.75s per 
                request), 15 Requests per Minute limit
            """
            models = self.cfg["speller"]["autocomplete"]["online"]["models"]
            genai.configure(
                api_key=self.cfg["speller"]["autocomplete"]["online"]["api_key"]
            )
            model_idx = self.cfg["speller"]["autocomplete"]["online"]["model_idx"]
            # instructions: str - instructions for the model to follow, can be used to guide the model to
            # generate specific content or avoid certain outputs
            instructions = self.cfg["speller"]["autocomplete"]["online"]["instructions"]
            model = genai.GenerativeModel(
                models[model_idx], system_instruction=instructions
            )
            return model

    def online_autocomplete(self, text: str) -> str:
        """
        use the generative AI model to generate the next word in the sentence
        """
        model = self.autocomplete_engine
        # temperature: float - temperature parameter for the model, higher values cause more randomness in the output
        temp = self.cfg["speller"]["autocomplete"]["online"]["temperature"]
        # output_length: int - maximum number of tokens in the output, longer outputs take longer to generate
        # current output_length is set to 20, or a maximum of around 10-15 words, though this is never reached
        output_length = self.cfg["speller"]["autocomplete"]["online"]["output_length"]
        # candidate_count: int - number of candidate outputs to generate, higher values may lead to better results
        # but take longer
        response = model.generate_content(
            text,
            generation_config=genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=output_length,
                temperature=temp,
            ),
        )
        return response.text.strip()

    def offline_autocomplete(self, text: str) -> str:
        """
        uses autocomplete module, based on bi-gram concept; given some current text, the model predicts the next word
        based on the most common word that follows the current word
        """
        # if the last character is a space, don't predict anything (wait for next character)
        if text[-1] == " ":
            return text
        autocomplete.load()
        words = text.split(" ")
        current_word = words[-1]

        # use 'the' as the previous word if no previous word is present
        if len(words) > 1:
            previous_word = words[-2]
        else:
            previous_word = "the"

        # try prediction, if no prediction is found, use 'the' as the previous word and try again
        result = autocomplete.predict(previous_word, current_word)
        if not result:
            result = autocomplete.predict("the", current_word)
            if not result:
                return text

        # if the current word is the only word in the text, capitalize the first letter of the prediction
        if len(words) == 1:
            return (result[0][0]).capitalize()
        else:
            return text[0 : len(text) - len(current_word)] + result[0][0]

    def start_autocomplete(self):
        """
        start the autocomplete process in its own thread, either online or offline based on the mode specified in the
        config
        """
        text = self.get_text_field("text")
        mode = self.cfg["speller"]["autocomplete"][
            "mode"
        ]  # mode is either "online" or "offline"

        # create task based on mode, start task in a new thread, update next_autocomplete with the result to be shown
        if mode == "online":

            def task():
                result = self.online_autocomplete(text)
                self.next_autocomplete = result
        else:

            def task():
                result = self.offline_autocomplete(text)
                self.next_autocomplete = result

        # start the task in a new thread
        thread = threading.Thread(target=task, daemon=True)
        thread.start()


def setup_speller(cfg: dict) -> Speller:
    # Setup speller
    speller = Speller(
        screen_resolution=cfg["speller"]["screen"]["resolution"],
        width_cm=cfg["speller"]["screen"]["width_cm"],
        distance_cm=cfg["speller"]["screen"]["distance_cm"],
        refresh_rate=cfg["speller"]["screen"]["refresh_rate_hz"],
        screen_id=cfg["speller"]["screen"]["id"],
        full_screen=cfg["speller"]["screen"]["full_screen"],
        background_color=cfg["speller"]["screen"]["background_color"],
        marker_stream_name=cfg["streams"]["marker_stream_name"],
        quit_controls=cfg["speller"]["controls"]["quit"],
        cfg=cfg,
        
    )
    print("DEBUG full_screen =", cfg["speller"]["screen"]["full_screen"])
    print("DEBUG screen_id =", cfg["speller"]["screen"]["id"])  
    ppd = speller.get_pixels_per_degree()

    # Add stimulus timing tracker at left top of the screen
    if cfg["speller"]["stt"]["enabled"]:
        x_pos = int(
            -cfg["speller"]["screen"]["resolution"][0] / 2
            + cfg["speller"]["stt"]["width_dva"] / 2 * ppd
        )
        y_pos = int(
            cfg["speller"]["screen"]["resolution"][1] / 2
            - cfg["speller"]["stt"]["height_dva"] / 2 * ppd
        )
        speller.add_key(
            name="stt",
            images=[
                Path(cfg["speller"]["images_dir"]) / f"{color}.png"
                for color in cfg["speller"]["stt"]["colors"]
            ],
            images_lower=[],
            size=(
                int(cfg["speller"]["stt"]["width_dva"] * ppd),
                int(cfg["speller"]["stt"]["height_dva"] * ppd),
            ),
            pos=(x_pos, y_pos),
        )

    # Add text field at the top of the screen containing spelled text
    if cfg["speller"]["stt"]["enabled"]:
        x_pos = int(cfg["speller"]["stt"]["width_dva"] * ppd / 2)
        x_size = int(
            cfg["speller"]["screen"]["resolution"][0]
            - cfg["speller"]["stt"]["width_dva"] * ppd
        )
    else:
        x_pos = 0
        x_size = int(cfg["speller"]["screen"]["resolution"][0])
    y_pos = int(
        cfg["speller"]["screen"]["resolution"][1] / 2
        - cfg["speller"]["text_fields"]["height_dva"] * ppd / 2
    )
    y_size = int(cfg["speller"]["text_fields"]["height_dva"] * ppd)

    speller.add_text_field(
        name="text",
        text="",
        size=(x_size, y_size),
        pos=(x_pos, y_pos),
        background_color=cfg["speller"]["text_fields"]["background_color"],
        text_color=(1.0, 1.0, 1.0),
    )

    # using the positions of the text field with an offset, add a text field for the autocompletion results
    if cfg["speller"]["autocomplete"]["enabled"]:
        speller.add_text_field(
            name="autocomplete_text",
            text="",
            size=(x_size, y_size),
            pos=(x_pos, y_pos - y_size),
            background_color=cfg["speller"]["text_fields"]["background_color"],
            text_color=(-0.7, -0.7, -0.7),
        )

    # Add text field at the bottom of the screen containing system messages
    x_pos = 0
    y_pos = int(
        -cfg["speller"]["screen"]["resolution"][1] / 2
        + cfg["speller"]["text_fields"]["height_dva"] * ppd / 2
    )
    speller.add_text_field(
        name="messages",
        text="",
        size=(
            cfg["speller"]["screen"]["resolution"][0],
            int(cfg["speller"]["text_fields"]["height_dva"] * ppd),
        ),
        pos=(x_pos, y_pos),
        background_color=cfg["speller"]["text_fields"]["background_color"],
        text_color=(1.0, 1.0, 1.0),
        alignment="center",
    )

    # Add keys
    keys_from_cfg = cfg["speller"]["keys"]["keys_upper"]
    for y in range(len(keys_from_cfg)):
        for x in range(len(keys_from_cfg[y])):
            x_pos = int(
                (x - len(keys_from_cfg[y]) / 2 + 0.5)
                * (
                    cfg["speller"]["keys"]["width_dva"]
                    + cfg["speller"]["keys"]["space_dva"]
                )
                * ppd
            )
            y_pos = int(
                -(y - len(keys_from_cfg) / 2)
                * (
                    cfg["speller"]["keys"]["height_dva"]
                    + cfg["speller"]["keys"]["space_dva"]
                )
                * ppd
                - cfg["speller"]["text_fields"]["height_dva"] * ppd
            )
            if y == 0 or y == 1:
                x_pos += int(0.25 * cfg["speller"]["keys"]["width_dva"] * ppd)
            elif y == 3 or y == 4:
                x_pos -= int(0.5 * cfg["speller"]["keys"]["width_dva"] * ppd)
            if keys_from_cfg[y][x] == "space":
                images = [
                    Path(cfg["speller"]["images_dir"]) / f"{color}.png"
                    for color in cfg["speller"]["keys"]["colors"]
                ]
                images_lower = images
            else:
                images = [
                    Path(cfg["speller"]["images_dir"])
                    / f"{keys_from_cfg[y][x]}_{color}.png"
                    for color in cfg["speller"]["keys"]["colors"]
                ]
                # if shifting is enabled, check if the key has a different shift key, if so, add the lowercase version
                # of the key
                if (
                    cfg["speller"]["keys"]["shift_enabled"]
                    and (keys_from_cfg[y][x])
                    != (cfg["speller"]["keys"]["keys_lower"][y][x])
                ):
                    # check if the key has a lower case version
                    if (
                        (keys_from_cfg[y][x]).isalpha()
                        and len(keys_from_cfg[y][x]) == 1
                    ):
                        images_lower = [
                            Path(cfg["speller"]["images_dir"])
                            / f"{keys_from_cfg[y][x]}_lower_{color}.png"
                            for color in cfg["speller"]["keys"]["colors"]
                        ]
                    else:  # special symbols i.e. 1 -> !, 2 -> @, etc.
                        images_lower = [
                            Path(cfg["speller"]["images_dir"])
                            / f"{cfg['speller']['keys']['keys_lower'][y][x]}_{color}.png"
                            for color in cfg["speller"]["keys"]["colors"]
                        ]
                else:  # keep upper and lowercase images the same
                    images_lower = images
            speller.add_key(
                name=keys_from_cfg[y][x],
                images=images,
                images_lower=images_lower,
                size=(
                    int(cfg["speller"]["keys"]["width_dva"] * ppd),
                    int(cfg["speller"]["keys"]["height_dva"] * ppd),
                ),
                pos=(x_pos, y_pos),
            )

    speller.init_highlights_with_zero()

    return speller


def create_key2seq_and_code2key(cfg: dict, phase: str) -> tuple[dict, dict]:
    codes_file = Path(cfg[phase]["codes_file"])

    # Setup code sequences from the correct phase
    codes = np.load(Path(cfg["speller"]["codes_dir"]) / codes_file)["codes"]
    codes = np.repeat(
        codes,
        int(
            cfg["speller"]["screen"]["refresh_rate_hz"]
            / cfg["speller"]["presentation_rate_hz"]
        ),
        axis=1,
    )

    # Optimal layout and subset:
    subset_layout_file = cfg["decoder"]["decoder_subset_layout_file"]
    if phase == "online" and len(subset_layout_file) > 0:
        # Fetch the subset and layout file location
        if os.path.isfile(subset_layout_file):
            with open(subset_layout_file, "r") as infile:
                data = json.load(infile)
                subset = np.array(data["subset"])
                layout = np.array(data["layout"])

            # Extra assertion check. The speller and decoder online/training code files should match.
            assert codes_file.name == data["codes_file"], (
                "The stimuli of the speller and decoder are not the same, please check."
            )

            # Set the loaded codes with subset and optimal layout
            # Note that this means that while i_code still refers to indices 0 trough to n_keys
            # The actual code that's placed there might for example originally be indices 59, 12, 0...
            # You can find the actual index values of the original code file by printing/comparing the optimal_layout
            # np array.
            codes = codes[subset, :]
            codes = codes[layout, :]
            logger.info("Stimulus subset and layout applied.")
        else:
            logger.info(f"Subset and layout file {subset_layout_file} not found.")
    else:
        logger.debug("No stimulus subset or layout applied.")

    key_to_sequence = dict()
    code_to_key = dict()
    i_code = 0
    keys_from_cfg = cfg["speller"]["keys"]["keys_upper"]
    for row in keys_from_cfg:
        for key in row:
            key_to_sequence[key] = codes[i_code, :].tolist()
            code_to_key[i_code] = key
            i_code += 1
    if cfg["speller"]["stt"]["enabled"]:
        key_to_sequence["stt"] = [1] + [0] * int(
            (1 + cfg["speller"]["timing"]["trial_s"])
            * cfg["speller"]["screen"]["refresh_rate_hz"]
        )

    return key_to_sequence, code_to_key


def run_speller_paradigm(
    phase: str = "training",
    config_path: Path = Path("./configs/speller.toml"),  # relative to the project root
) -> int:
    """
    Run the speller in a particular phase (training or online).

    Parameters
    ----------
    phase: str (default: "training")
        The phase of the speller being either training or online. During training, the user is cued to attend to a
        random target key every trail. During online, the user attends to their target, while their EEG is decoded and
        the decoded key is used to perform an action (e.g., add a symbol to a sentence, backspace, etc.). In the online
        phase, the speller will continuously query an LSL marker stream to look for a potential decoding result from
        the decoder module. If the stream contains a marker `speller_select <key_idx>`, the speller will
        stop the presentation and will show the selected symbol.
    config_path: Path (default: "./configs/speller.toml")
        The path to the configuration file containing session specific hyperparameters for the speller setup.

    Returns
    -------
    flag: int
        Whether the process ran without errors or with.
    """
    cfg = toml.load(config_path)
    speller = setup_speller(cfg)
    logger.info(f'[CHECK] Speller setup complete for phase "{phase}".')

    if phase != "training":
        speller.connect_to_decoder_lsl_stream()
    speller.create_marker_lsl_stream()

    key_to_sequence, code_to_key = create_key2seq_and_code2key(cfg, phase)
    speller.key_map = code_to_key
    n_classes = len(code_to_key)

    # Wait to start run
    logger.info("Waiting for button press to start")
    speller.set_text_field(name="messages", text="Press button to start.")
    event.waitKeys(keyList=cfg["speller"]["controls"]["continue"])
    speller.set_text_field(name="messages", text="")

    # Log info
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    speller.log(
        marker=f"version;python={python_version};psychopy={psychopy.__version__}"
    )
    speller.log(marker=f"setup;codes={key_to_sequence};labels={code_to_key}")

    # Start run
    logger.info("Starting")
    speller.log(marker="start_run")
    speller.set_text_field(name="messages", text="Starting...")
    speller.run(sequences=speller.highlights, duration=5.0)
    speller.set_text_field(name="messages", text="")

    # Loop trials
    n_trials = cfg[phase]["n_trials"]
    completed_online_trials = 0
    online_true_targets = [
        str(target)
        for target in cfg.get("online_accuracy", {}).get("true_targets", [])
    ]
    for i_trial in range(n_trials):
        logger.info(f"Initiating trial {1 + i_trial}/{n_trials}")

        if phase == "training":
            # Set a random target
            target = random.randint(0, n_classes - 1)
            target_key = code_to_key[target]

            # Cue
            logger.info(f"Cueing {target_key} ({target})")
            speller.highlights[target_key] = [-2]
            speller.run(
                sequences=speller.highlights,
                duration=cfg["speller"]["timing"]["cue_s"],
                start_marker=f"{cfg['speller']['markers']['cue_start']};label={target};key={target_key}",
                stop_marker=cfg["speller"]["markers"]["cue_stop"],
            )
            speller.highlights[target_key] = [0]

        if phase == "online" and speller.cfg["speller"]["autocomplete"]["enabled"]:
            speller.set_text_field(
                name="autocomplete_text", text=speller.next_autocomplete
            )

        # Trial
        logger.info("Starting stimulation")
        if phase == "online":
            speller.current_trial_id = i_trial + 1
            speller.accept_decoder_output = True
            speller.decoding_event_seen = False
            speller.last_selected_key_idx = None
            speller.last_selected_position = None
            speller.last_selected_label = None
            n_drained = speller.drain_decoder_stream()
            if n_drained > 0:
                logger.info(
                    f"[CHECK] Cleared {n_drained} stale decoder sample(s) before trial "
                    f"{1 + i_trial}/{n_trials}."
                )
            logger.info(
                f"[CHECK] Trial {speller.current_trial_id}/{n_trials}: arming decoder "
                "before flashing."
            )
            speller.send_decoder_marker(cfg["speller"]["markers"]["trial_start"])
            decoder_arm_wait_s = cfg["speller"]["timing"].get("decoder_arm_wait_s", 2.0)
            decoder_armed = speller.wait_for_decoder_arm(
                    duration=decoder_arm_wait_s,
                    retry_marker=cfg["speller"]["markers"]["trial_start"],
            )
            if decoder_armed:
                speller.skip_next_udp_trial_start = True
            else:
                logger.warning(
                    f"[CHECK] Trial {speller.current_trial_id}/{n_trials}: "
                    "decoder arm acknowledgement was not received; continuing trial "
                    "without a duplicate UDP start marker. The end-of-trial "
                    "force_decision marker will request the real rCCA output."
                )
                speller.skip_next_udp_trial_start = True
        speller.run(
            sequences=key_to_sequence,
            duration=cfg["speller"]["timing"]["trial_s"],
            start_marker=f"{cfg['speller']['markers']['trial_start']}",
            stop_marker=cfg["speller"]["markers"]["trial_stop"],
            start_marker_on_flip=phase != "online",
            check_decoder_output=phase == "online",
        )
        if phase == "online" and not speller.decoding_event_seen:
            speller.request_decoder_force_decision()
            decoder_wait_s = cfg["speller"]["timing"].get("decoder_wait_s", 2.0)
            logger.info(
                f"[CHECK] Trial {speller.current_trial_id}/{n_trials}: waiting "
                f"{decoder_wait_s:.1f}s after flashing for matching decoder output."
            )
            speller.wait_for_decoding_event(
                duration=decoder_wait_s,
                retry_force_decision=True,
            )
            if not speller.decoding_event_seen:
                emergency_wait_s = cfg["speller"]["timing"].get(
                    "decoder_emergency_wait_s", 0.0
                )
                if emergency_wait_s > 0:
                    logger.error(
                        f"[CHECK] Trial {speller.current_trial_id}/{n_trials}: "
                        "no matching rCCA decoder output received during normal wait; "
                        "continuing to request a real rCCA decision before ending this trial."
                    )
                    speller.wait_for_decoding_event(
                        duration=emergency_wait_s,
                        retry_force_decision=True,
                    )
            if not speller.decoding_event_seen:
                logger.error(
                    f"[CHECK] Trial {speller.current_trial_id}/{n_trials}: "
                    f"no real rCCA decoder output within {decoder_wait_s:.1f}s; "
                    "recording this trial as no-output and continuing."
                )
        if phase == "online":
            true_target = (
                online_true_targets[i_trial]
                if i_trial < len(online_true_targets)
                else None
            )
            speller.record_online_accuracy_trial(
                trial_id=i_trial + 1,
                true_target=true_target,
            )
            completed_online_trials += 1
            speller.accept_decoder_output = False

        # Inter-trial time
        logger.info("Inter-trial interval")
        speller.run(
            sequences=speller.highlights,
            duration=cfg["speller"]["timing"]["iti_s"],
            start_marker=f"{cfg['speller']['markers']['iti_start']}",
            stop_marker=cfg["speller"]["markers"]["iti_stop"],
            check_decoder_output=False,
        )
        if phase == "online":
            speller.current_trial_id = None

        if phase == "online" and speller.get_text_field("text").endswith(
            cfg["speller"]["quit_phrase"]
        ):
            break

    if phase == "online" and completed_online_trials == n_trials:
        plot_delay_s = float(cfg.get("online_accuracy", {}).get("plot_delay_s", 0.0))
        if plot_delay_s > 0:
            logger.info(
                f"[CHECK] Waiting {plot_delay_s:.1f}s before opening online accuracy plot "
                "so final trial feedback remains visible."
            )
            delay_deadline = time.time() + plot_delay_s
            while time.time() < delay_deadline:
                if len(event.getKeys(keyList=cfg["speller"]["controls"]["quit"])) > 0:
                    speller.quit()
                    return
                speller.window.flip()
                time.sleep(1.0 / max(float(speller.refresh_rate), 1.0))
        speller.save_online_accuracy_plot()
    elif phase == "online":
        logger.warning(
            f"[CHECK] Online run ended after {completed_online_trials}/{n_trials} "
            "trials; final accuracy plot was not saved because the full run did not finish."
        )

    # Wait to stop
    logger.info("Waiting for button press to stop")
    speller.set_text_field(name="messages", text="Press button to stop.")
    event.waitKeys(keyList=cfg["speller"]["controls"]["continue"])
    speller.set_text_field(name="messages", text="")

    # Stop run
    logger.info("Stopping")
    speller.log(marker="stop_run")
    speller.set_text_field(name="messages", text="Stopping...")
    speller.run(
        sequences=speller.highlights,
        duration=cfg["speller"]["timing"].get("stop_s", 5.0),
    )
    speller.set_text_field(name="messages", text="")
    speller.quit()

    return 0


# make this the cli starting point
def cli_run(
    phase: str = "training",
    config_path: Path = Path("./configs/speller.toml"),  # relative to the project root
    log_level: int = 30,
) -> None:
    """
    Run the speller in a particular phase (training or online).

    Parameters
    ----------
    phase: str (default: "training")
        The phase of the speller being either training or online. During training, the user is cued to attend to a
        random target key every trail. During online, the user attends to their target, while their EEG is decoded and
        the decoded key is used to perform an action (e.g., add a symbol to a sentence, backspace, etc.). In the online
        phase, the speller will continuously query an LSL marker stream to look for a potential decoding result from
        the decoder module. If the stream contains a marker `speller_select <key_idx>`, the speller will
        stop the presentation and will show the selected symbol.
    config_path: Path (default: "./configs/speller.toml")
        The path to the configuration file containing session specific hyperparameters for the speller setup.
    log_level : int (default: 30)
        The logging level to use.
    """

    # activate the console logging if started via CLI
    logger = get_logger("cvep-speller", add_console_handler=True)
    logger.setLevel(log_level)

    run_speller_paradigm(phase=phase, config_path=config_path)


if __name__ == "__main__":
    Fire(cli_run)
