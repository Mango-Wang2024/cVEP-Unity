import socket
import threading

from pylsl import StreamInfo, StreamOutlet

from cvep_speller.utils.logging import logger


class UdpToLslMarkerBridge:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9098,
        stream_name: str = "cvep-speller-stream",
    ) -> None:
        self.host = host
        self.port = port
        self.stream_name = stream_name
        self.socket: socket.socket | None = None
        self.outlet: StreamOutlet | None = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()

    def start(self) -> int:
        if self.thread is not None and self.thread.is_alive():
            logger.info(
                f'[CHECK] Marker stream "{self.stream_name}" is already available '
                "for Lab Recorder."
            )
            return 0

        info = StreamInfo(
            name=self.stream_name,
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id=self.stream_name,
        )
        self.outlet = StreamOutlet(info)

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.settimeout(0.1)
        except OSError as err:
            logger.error(
                f"Could not start Unity marker LSL bridge on "
                f"{self.host}:{self.port}: {err}"
            )
            self.socket = None
            self.outlet = None
            return 1

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logger.info(
            f'[CHECK] Unity marker LSL bridge started: UDP {self.host}:{self.port} '
            f'-> LSL "{self.stream_name}".'
        )
        logger.info(
            f'[CHECK] Marker stream "{self.stream_name}" is ready. In Lab Recorder, '
            "click Update and select it together with obci_eeg1."
        )
        return 0

    def stop(self) -> None:
        self.stop_event.set()

        if self.socket is not None:
            self.socket.close()
            self.socket = None

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)

        self.thread = None
        self.outlet = None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if self.socket is None or self.outlet is None:
                return

            try:
                data, _ = self.socket.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                return

            marker = data.decode("utf-8").strip()
            if not marker:
                continue

            self.outlet.push_sample([marker])
            logger.info(f'[CHECK] Unity marker bridged to LSL: "{marker}"')
