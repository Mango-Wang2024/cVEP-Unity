from pathlib import Path

import toml


ROOT_DIR = Path("cvep_speller_env")
DATA_DIR = ROOT_DIR / "data"
DECODER_CFG = ROOT_DIR / "dp-cvep-decoder" / "configs" / "decoder.toml"
CAPFILE = ROOT_DIR / "dp-cvep-decoder" / "cvep_decoder" / "caps" / "openbci8_headset.loc"

DATA_STREAM_NAME = "obci_eeg1"
HEADSET_CHANNELS = [0, 1, 2, 3, 4, 5, 6, 7]

CAPFILE_CONTENT = """1	      90	       0	      Cz
2	     162	 0.51111	      O2
3	     108	 0.45278	     TP8
4	       0	 0.51111	     FPz
5	    -162	 0.51111	      O1
6	     180	 0.51111	      Oz
7	     180	 0.25556	      Pz
8	    -108	 0.45278	     TP7"""


def main() -> None:
    if not DECODER_CFG.exists():
        raise FileNotFoundError(f"Decoder config not found: {DECODER_CFG}")

    CAPFILE.write_text(CAPFILE_CONTENT)

    cfg = toml.load(DECODER_CFG)
    cfg["streams"]["data_stream_name"] = DATA_STREAM_NAME
    cfg["data"]["data_root"] = str(DATA_DIR.resolve())
    cfg["data"]["selected_channels"] = HEADSET_CHANNELS
    cfg["data"]["capfile"] = str(CAPFILE.resolve())

    toml.dump(cfg, DECODER_CFG.open("w"))

    print(f"Updated {DECODER_CFG}")
    print(f"Selected Cyton channels: {HEADSET_CHANNELS}")
    print(f"Capfile: {CAPFILE.resolve()}")


if __name__ == "__main__":
    main()
