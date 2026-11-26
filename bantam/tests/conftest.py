"""Shared pytest markers. CI runs: pytest -m "not slow and not network and not gpu"."""
import pytest

MARKERS = {
    "slow": "takes more than a few seconds (builds, end-to-end runs)",
    "network": "needs internet access (Hugging Face Hub, PyPI)",
    "gpu": "needs a CUDA device; skipped automatically without one",
    "dist": "spawns several processes (torch.distributed, gloo)",
}


def pytest_configure(config):
    for name, doc in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {doc}")


def pytest_collection_modifyitems(config, items):
    gpu_items = [it for it in items if "gpu" in it.keywords]
    if not gpu_items:
        return
    import torch
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="needs a CUDA device")
        for it in gpu_items:
            it.add_marker(skip)
