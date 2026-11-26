"""Command line: `bantam <command> [options]` (also `python -m bantamtrain <command>`).

    bantam pretrain      train a ZoomGPT language model at a fixed FLOP budget
    bantam finetune      LoRA / QLoRA / full fine-tuning of a Hugging Face causal LM
    bantam weather       train a ZoomViT 6 h ERA5 forecaster
    bantam prepare-text  Hugging Face text dataset -> tokenizer + token files (uint16/uint32)
    bantam pack-text     dedup + TokPack a tokenized corpus
    bantam prepare-era5  download the WeatherBench2 5.625 deg ERA5 subset
    bantam pack-era5     ERA5 fields -> error-bounded GridPack
    bantam memory        single-GPU memory / time scoreboard
    bantam generate      sample text from a trained ZoomGPT
    bantam chat          chat with a (fine-tuned) Hugging Face model
    bantam doctor        environment report for bug reports (--json)

`bantam <command> --help` lists a command's options.
"""
import importlib
import importlib.util
import inspect
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
    "generate": "bantamtrain.cli.generate",
    "chat": "bantamtrain.cli.chat",
    "doctor": "bantamtrain.cli.doctor",
}
# optional dependency (top-level module) -> the pip extra that installs it (None: pip install it)
EXTRAS = {
    "pyarrow": "data", "tokenizers": "data", "huggingface_hub": "data",
    "transformers": "finetune", "datasets": "finetune", "sentencepiece": "finetune",
    "safetensors": "finetune", "peft": None, "accelerate": None,
    "xarray": "weather", "zarr": "weather", "fsspec": "weather", "aiohttp": "weather",
    "gcsfs": None, "matplotlib": "test",
}


def _missing_extra(e, module):
    """Friendly message for a ModuleNotFoundError raised by a command, or None to re-raise."""
    name = e.name or ""
    if name and (module == name or module.startswith(name + ".")):
        return f"this installation has no `{module}` (upgrade bantamtrain)"
    top = name.split(".")[0]
    if top not in EXTRAS or _installed(top):    # a missing submodule is a version problem: re-raise
        return None
    how = f"'bantamtrain[{EXTRAS[top]}]'" if EXTRAS[top] else top
    return f"needs `{top}`, which is not installed: pip install {how}"


def _installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ValueError, ImportError):            # blocked or broken (sys.modules[m] = None)
        return False


def _run(mod, rest):
    """mod.main(argv) when main takes an argument, else mod.main() reading sys.argv."""
    try:
        takes_argv = bool(inspect.signature(mod.main).parameters)
    except (TypeError, ValueError):
        takes_argv = False
    return mod.main(rest) if takes_argv else mod.main()


def main(argv=None):
    """Run a command and return its exit code (int); sys.argv is restored afterwards."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
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
    old_argv = sys.argv
    sys.argv = [f"bantam {cmd}"] + argv[1:]        # argparse takes its prog name from argv[0]
    try:
        rc = _run(importlib.import_module(COMMANDS[cmd]), argv[1:])
    except ModuleNotFoundError as e:
        msg = _missing_extra(e, COMMANDS[cmd])
        if msg is None:
            raise
        print(f"bantam {cmd}: {msg}", file=sys.stderr)
        return 1
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        # a mistyped --data / --jsonl / --adapter / --load path: one line, as argparse's usage
        # errors (exit 2), not a traceback
        print(f"bantam {cmd}: error: {e}", file=sys.stderr)
        return 2
    except SystemExit as e:                         # argparse --help / usage errors, sys.exit()
        rc = e.code
    finally:
        sys.argv = old_argv
    if rc is None or isinstance(rc, bool):
        return int(bool(rc))
    if isinstance(rc, int):
        return rc
    print(rc, file=sys.stderr)                      # sys.exit("message") convention
    return 1
