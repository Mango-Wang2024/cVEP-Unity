import socket
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

        return self._send_command("training")

    def online(self) -> int:
        self.marker_bridge.stop()
        return self._send_command(self._online_command_from_decoder_config())

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
                sock.sendto(payload, (self.host, int(self.port)))
        except OSError as err:
            logger.error(
                f"Could not send Unity frontend command '{command}' to "
                f"{self.host}:{self.port}: {err}"
            )
            return 1

        logger.info(
            f"Sent Unity frontend command '{command}' to {self.host}:{self.port}. "
            "Make sure Unity is in Play Mode, then press c in Unity."
        )
        return 0
