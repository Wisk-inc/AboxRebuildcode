"""Command line: `bantam <command> [options]` (also `python -m bantamtrain <command>`).

    bantam pretrain      train a ZoomGPT language model at a fixed FLOP budget
    bantam finetune      LoRA / QLoRA / full fine-tuning of a Hugging Face causal LM
    bantam weather       train a ZoomViT 6 h ERA5 forecaster
    bantam prepare-text  Hugging Face text dataset -> tokenizer + uint16 token files
    bantam pack-text     dedup + TokPack a tokenized corpus
    bantam prepare-era5  download the WeatherBench2 5.625 deg ERA5 subset
    bantam pack-era5     ERA5 fields -> error-bounded GridPack
    bantam memory        single-GPU memory / time scoreboard
"""
import importlib
import sys

COMMANDS = {
    "pretrain": "bantamtrain.cli.pretrain",
    "finetune": "bantamtrain.finetune",
    "weather": "bantamtrain.cli.weather",
    "prepare-text": "bantamtrain.cli.prepare_text",
    "pack-text": "bantamtrain.cli.pack_text",
    "prepare-era5": "bantamtrain.cli.prepare_era5",
    "pack-era5": "bantamtrain.cli.pack_era5",
    "memory": "bantamtrain.memory",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] in ("-V", "--version"):
        from .. import __version__
        print(__version__)
        return 0
    cmd = argv[0]
    if cmd not in COMMANDS:
        print(f"bantam: unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
        return 2
    mod = importlib.import_module(COMMANDS[cmd])
    sys.argv = [f"bantam {cmd}"] + argv[1:]
    return mod.main()
