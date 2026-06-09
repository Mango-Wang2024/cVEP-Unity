import os
from functools import partial

from dareplane_utils.default_server.server import DefaultServer
from fire import Fire

from cvep_speller.speller import run_speller_paradigm
from cvep_speller.unity_frontend import UnityFrontendController
from cvep_speller.utils.logging import logger


def main(
    port: int = 8080,
    ip: str = "127.0.0.1",
    loglevel: int = 10,
    frontend: str | None = None,
    unity_host: str = "127.0.0.1",
    unity_port: int = 9110,
    unity_marker_bridge_host: str = "127.0.0.1",
    unity_marker_bridge_port: int = 9098,
) -> int:
    logger.setLevel(loglevel)

    frontend = (frontend or os.environ.get("CVEP_SPELLER_FRONTEND", "psychopy")).lower()
    unity = UnityFrontendController(
        host=unity_host,
        port=unity_port,
        marker_bridge_host=unity_marker_bridge_host,
        marker_bridge_port=unity_marker_bridge_port,
    )

    # Primary commands
    if frontend == "unity":
        logger.info("Using Unity frontend for TRAINING and ONLINE commands.")
        pcommand_map = {
            "TRAINING": unity.training,
            "ONLINE": unity.online,
        }
    else:
        logger.info("Using PsychoPy frontend for TRAINING and ONLINE commands.")
        pcommand_map = {
            "TRAINING": partial(run_speller_paradigm, phase="training"),
            "ONLINE": partial(run_speller_paradigm, phase="online"),
        }

    pcommand_map.update(
        {
            "UNITY TRAINING": unity.training,
            "UNITY ONLINE": unity.online,
        }
    )

    # Setup server
    server = DefaultServer(
        port, ip=ip, pcommand_map=pcommand_map, name="cvep_speller_server"
    )
    server.init_server()
    server.start_listening()

    return 0


if __name__ == "__main__":
    Fire(main)
