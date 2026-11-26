"""0.3.0 release review: a core-only install imports no optional dependency, `bantam
--help` / `--version` start without torch, the sdist leaves out tests/, and the finetune
docstring example's --examples_per_step changes --accum."""
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bantamtrain import cli  # noqa: E402

ENV = {**os.environ, "OMP_NUM_THREADS": "1"}
# every optional dependency (pyproject extras, bantamtrain.cli.EXTRAS) except fsspec, which
# torch itself installs
OPTIONAL = sorted(set(cli.EXTRAS) - {"fsspec"} | {"psutil"})

# Runs in a fresh interpreter. The optional dependencies are blocked as if not installed,
# and every attempt to import one from bantamtrain's own code is recorded (also an attempt
# inside try/except, which a blocked import alone would not reveal).
CORE_ONLY = textwrap.dedent("""
    import io, json, os, sys, contextlib
    sys.path.insert(0, ROOT)
    tried = []

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] not in OPTIONAL:
                return None
            f = sys._getframe(1)
            while f is not None and f.f_code.co_filename.startswith("<frozen"):
                f = f.f_back
            if f is not None and os.sep + "bantamtrain" + os.sep in f.f_code.co_filename:
                tried.append([name, f.f_code.co_filename])
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    sys.meta_path.insert(0, Block())
    import bantamtrain
    from bantamtrain import cli
    out = {"torch_after_import": "torch" in sys.modules}
    with contextlib.redirect_stdout(io.StringIO()):
        out["rc_version"] = cli.main(["--version"])
        out["rc_help"] = cli.main(["--help"])
    out["torch_after_help"] = "torch" in sys.modules
    import torch
    ftz = lambda: float(torch.tensor([1e-39]) * 1.0) == 0.0
    out["ftz_before"] = ftz()
    out["rc"] = {}
    late = ("pretrain", "weather")      # documented: these turn flush-to-zero on at import
    cmds = [c for c in cli.COMMANDS if c not in late] + list(late)
    for c in cmds:
        if c == late[0]:
            import bantamtrain.finetune, bantamtrain.lora, bantamtrain.qweights
            import bantamtrain.losses, bantamtrain.weightpack, bantamtrain.memory
            import bantamtrain.data, bantamtrain.models, bantamtrain.cli.recipes
            out["ftz_before_pretrain"] = ftz()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            out["rc"][c] = cli.main([c, "--help"])
    out["tried"] = tried
    out["loaded"] = sorted(m for m in OPTIONAL if m in sys.modules)
    print(json.dumps(out))
""")


def test_core_only_install_imports_no_optional_dependency():
    code = f"ROOT = {ROOT!r}\nOPTIONAL = {OPTIONAL!r}\n" + CORE_ONLY
    p = subprocess.run([sys.executable, "-I", "-c", code], env=ENV, capture_output=True,
                       text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-3000:]
    import json
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["tried"] == [] and out["loaded"] == [], out
    assert out["rc"] == {c: 0 for c in cli.COMMANDS}, out["rc"]
    assert out["rc_version"] == 0 and out["rc_help"] == 0
    # `bantam --help` / `--version` (bantamtrain/__init__ imports zoom lazily) need no torch
    assert not out["torch_after_import"] and not out["torch_after_help"]
    # only importing the pretrain / weather command modules touches flush-to-zero
    assert not out["ftz_before"] and not out["ftz_before_pretrain"]


def test_package_attributes_are_imported_on_first_use():
    import bantamtrain
    from bantamtrain import zoom
    from bantamtrain import ZoomSchedule, lr_at, SlotMerge, SlotUnmerge
    assert (ZoomSchedule, lr_at, SlotMerge, SlotUnmerge) == \
        (zoom.ZoomSchedule, zoom.lr_at, zoom.SlotMerge, zoom.SlotUnmerge)
    assert set(bantamtrain.__all__) == {"ZoomSchedule", "lr_at", "SlotMerge", "SlotUnmerge"}
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        bantamtrain.nope


def test_finetune_docstring_example_changes_accum():
    """Its --examples_per_step must not be a no-op: at the default --batch 4 and ~3.4 Dolly
    examples per 512-token window, 8 examples per step rounds to --accum 1, the default."""
    from bantamtrain import finetune as ft
    doc = ft.__doc__
    cmd = doc[doc.index("python -m bantamtrain.finetune"):].split("\n\n")[0].replace("\\\n", " ").split()
    assert "--batch" in cmd and "--seq" not in cmd and "--accum" not in cmd
    batch, n = int(cmd[cmd.index("--batch") + 1]), int(cmd[cmd.index("--examples_per_step") + 1])
    per_window = 3.4                     # README: SmolLM2 tokenizer, Dolly, --seq 512
    accum = max(1, round(n / (batch * per_window)))    # finetune.main's rule, one process
    assert accum > 1


@pytest.mark.slow
def test_sdist_leaves_out_tests(tmp_path):
    """setuptools adds tests/test*.py to an sdist by default, without conftest.py and the
    experiments/ and results/ they read; MANIFEST.in prunes them."""
    pytest.importorskip("build")
    src = tmp_path / "src"
    ign = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info")
    for d in ("bantamtrain", "packaging", "tests"):
        shutil.copytree(os.path.join(ROOT, d), src / d, ignore=ign)
    for f in ("pyproject.toml", "MANIFEST.in", "README.md"):
        shutil.copy(os.path.join(ROOT, f), src / f)
    p = subprocess.run([sys.executable, "-m", "build", "--sdist", "--no-isolation", "--outdir",
                        str(tmp_path / "dist"), str(src)], env=ENV, capture_output=True,
                       text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    (sdist,) = (tmp_path / "dist").glob("*.tar.gz")
    with tarfile.open(sdist) as t:
        names = t.getnames()
    top = {n.split("/")[1] for n in names if n.count("/") >= 1}
    assert "bantamtrain" in top and "packaging" in top and "pyproject.toml" in top
    assert "tests" not in top, sorted(top)
    assert any(n.endswith("/bantamtrain/cli/recipes.py") for n in names)
