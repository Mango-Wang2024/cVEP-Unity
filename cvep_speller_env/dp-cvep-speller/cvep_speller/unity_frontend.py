import socket
import time
from pathlib import Path

import toml
from cvep_speller.lsl_marker_bridge import UdpToLslMarkerBridge
from cvep_speller.utils.logging import logger


class UnityFrontendController:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9110,
        marker_bridge_host: str = "127.0.0.1",
        marker_bridge_port: int = 9098,
        marker_stream_name: str = "cvep-speller-stream",
        decoder_config_file: Path | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.decoder_config_file = decoder_config_file or (
            Path(__file__).resolve().parents[2]
            / "dp-cvep-decoder"
            / "configs"
            / "decoder_unity.toml"
        )
        self.marker_bridge = UdpToLslMarkerBridge(
            host=marker_bridge_host,
            port=marker_bridge_port,
            stream_name=marker_stream_name,
        )

    def training(self) -> int:
        bridge_status = self.marker_bridge.start()
        if bridge_status != 0:
            return bridge_status

        status = self._send_command("training")
        if status == 0:
            self._log_training_recording_ready()
        return status

    def online(self) -> int:
        bridge_status = self.marker_bridge.start()
        if bridge_status != 0:
            return bridge_status

        status = self._send_command(self._online_command_from_decoder_config())
        if status == 0:
            self._log_online_recording_ready()
        return status

    def _online_command_from_decoder_config(self) -> str:
        try:
            cfg = toml.load(self.decoder_config_file)
            training_mode = str(
                cfg.get("decoder", {}).get("training_mode", "calibration")
            ).strip().lower()
        except Exception as err:
            logger.warning(
                f"Could not read decoder config '{self.decoder_config_file}' "
                f"for Unity online mode: {err}. Falling back to online_n_train."
            )
            training_mode = "calibration"

        if training_mode == "zero":
            return "online_zero_train"

        return "online_n_train"

    def _send_command(self, command: str) -> int:
        payload = command.encode("utf-8")

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                for _ in range(3):
                    sock.sendto(payload, (self.host, int(self.port)))
                    time.sleep(0.05)
        except OSError as err:
            logger.error(
                f"Could not send Unity frontend command '{command}' to "
                f"{self.host}:{self.port}: {err}"
            )
            return 1

        logger.info(
            f"[CHECK] Sent Unity frontend command '{command}' to "
            f"{self.host}:{self.port} with 3 UDP attempts. "
            "Make sure Unity is in Play Mode, then press c in Unity."
        )
        return 0

    def _log_training_recording_ready(self) -> None:
        logger.info(
            "[CHECK] UNITY TRAINING is ready for recording. In Lab Recorder, click "
            "Update and confirm both streams are selected: obci_eeg1 and "
            "cvep-speller-stream. Start Lab Recorder now, then press C in Unity."
        )

    def _log_online_recording_ready(self) -> None:
        logger.info(
            "[CHECK] UNITY ONLINE marker stream is ready: cvep-speller-stream "
            "should now appear in Lab Recorder. Click Update and select both "
            "obci_eeg1 and cvep-speller-stream before pressing C in Unity if "
            "you want an online XDF recording."
        )
