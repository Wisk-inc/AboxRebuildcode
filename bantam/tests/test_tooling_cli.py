"""`bantam` dispatcher, `bantam doctor`, packaging metadata and the release script."""
import importlib.util
import json
import os
import re
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import bantamtrain  # noqa: E402
from bantamtrain import cli  # noqa: E402

ENV = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": ROOT}


def _toml():
    try:
        import tomllib
    except ImportError:                    # Python 3.10
        tomllib = pytest.importorskip("tomli")
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)


def _floor(reqs, name):
    for r in reqs:
        m = re.fullmatch(rf"{name}\s*>=\s*([\d.]+)", r.strip())
        if m:
            return tuple(int(x) for x in m.group(1).split("."))
    return None


@pytest.mark.parametrize("cmd", sorted(cli.COMMANDS))
def test_every_command_has_help(cmd, capsys):
    rc = cli.main([cmd, "--help"])
    out, err = capsys.readouterr()
    if importlib.util.find_spec(cli.COMMANDS[cmd]) is None:    # written by another group
        assert rc == 1 and "this installation has no" in err
    elif rc == 1:                                  # an extra this environment lacks (CI: weather)
        assert "which is not installed: pip install" in err
    else:
        assert rc == 0 and "usage: bantam " + cmd in out


def test_main_restores_argv_and_returns_int(capsys):
    before = list(sys.argv)
    assert cli.main(["memory", "--params", "1e9"]) == 0
    assert sys.argv == before
    assert cli.main(["memory", "--no_such_flag"]) == 2          # argparse error -> its code
    assert sys.argv == before
    assert cli.main([]) == 0 and cli.main(["--help"]) == 0
    assert cli.main(["no-such-command"]) == 2
    capsys.readouterr()


def test_exit_codes_from_commands(monkeypatch, capsys):
    mod = types.ModuleType("bantam_fake_cmd")
    calls = {}

    def main(argv=None):
        calls["argv"], calls["sys.argv"] = argv, list(sys.argv)
        return {"ok": None, "fail": 3, "msg": "boom"}.get(argv[0])

    mod.main = main
    monkeypatch.setitem(sys.modules, "bantam_fake_cmd", mod)
    monkeypatch.setitem(cli.COMMANDS, "fake", "bantam_fake_cmd")
    assert cli.main(["fake", "ok"]) == 0
    assert calls == {"argv": ["ok"], "sys.argv": ["bantam fake", "ok"]}
    assert cli.main(["fake", "fail"]) == 3
    assert cli.main(["fake", "msg"]) == 1 and "boom" in capsys.readouterr().err
    mod.main = lambda: sys.exit(None if sys.argv[1:] == ["x"] else 4)   # legacy main() reads sys.argv
    assert cli.main(["fake", "x"]) == 0 and cli.main(["fake", "y"]) == 4


def test_missing_optional_dependency_gives_install_hint(monkeypatch, capsys):
    mod = types.ModuleType("bantam_fake_cmd")

    def main(argv=None):
        import pyarrow_not_installed  # noqa: F401

    mod.main = main
    monkeypatch.setitem(sys.modules, "bantam_fake_cmd", mod)
    monkeypatch.setitem(cli.COMMANDS, "fake", "bantam_fake_cmd")
    with pytest.raises(ModuleNotFoundError):                      # not an optional extra
        cli.main(["fake"])
    monkeypatch.setitem(cli.EXTRAS, "pyarrow_not_installed", "data")
    assert cli.main(["fake"]) == 1
    assert "pip install 'bantamtrain[data]'" in capsys.readouterr().err
    monkeypatch.setitem(cli.EXTRAS, "pyarrow_not_installed", None)    # in no extra
    assert cli.main(["fake"]) == 1
    assert capsys.readouterr().err.rstrip().endswith("pip install pyarrow_not_installed")

    def main_submodule(argv=None):                    # installed package, missing submodule
        import json.no_such_submodule  # noqa: F401

    mod.main = main_submodule
    monkeypatch.setitem(cli.EXTRAS, "json", "data")
    with pytest.raises(ModuleNotFoundError):
        cli.main(["fake"])


def test_help_without_optional_extras():
    """Core-only install: every --help works or names the extra to install; no traceback."""
    blocked = sorted(cli.EXTRAS)
    code = ("import sys\n"
            f"for m in {blocked!r}: sys.modules[m] = None\n"
            "from bantamtrain.cli import main, COMMANDS\n"
            "for c in COMMANDS: print(c, main([c, '--help']), flush=True)\n")
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=ENV, capture_output=True,
                       text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    assert "Traceback" not in p.stderr
    rcs = dict(line.split()[:2] for line in p.stdout.splitlines() if line.split()[:1] and
               line.split()[0] in cli.COMMANDS and len(line.split()) == 2)
    assert set(rcs) == set(cli.COMMANDS)
    for c, rc in rcs.items():
        assert rc in ("0", "1"), (c, rc)
        if rc == "1":
            assert f"bantam {c}: " in p.stderr
    for c in ("pretrain", "weather", "pack-text", "pack-era5", "memory", "doctor"):
        assert rcs[c] == "0"


def test_console_entry_point_and_version():
    p = subprocess.run([sys.executable, "-m", "bantamtrain", "--version"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout.strip() == bantamtrain.__version__
    p = subprocess.run([sys.executable, "-m", "bantamtrain", "nope"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 2


def test_version_is_single_sourced():
    t = _toml()
    assert "version" not in t["project"] and "version" in t["project"]["dynamic"]
    assert t["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "bantamtrain.__version__"}
    try:
        from importlib.metadata import PackageNotFoundError, version
        assert version("bantamtrain") == bantamtrain.__version__
    except PackageNotFoundError:          # running from a checkout that is not installed
        pass


def test_declared_minimums():
    t = _toml()
    assert _floor(t["build-system"]["requires"], "setuptools") >= (77,)   # PEP 639 license
    assert _floor(t["project"]["dependencies"], "torch") >= (2, 4)       # torch.OutOfMemoryError
    extras = t["project"]["optional-dependencies"]
    assert _floor(extras["finetune"], "transformers") >= (4, 56)          # from_pretrained(dtype=)
    test = " ".join(extras["test"])
    for mod in ("pytest", "transformers", "peft", "tokenizers", "pyarrow", "matplotlib"):
        assert mod in test
    for f in (t["project"]["readme"], *t["project"]["license-files"]):
        assert os.path.exists(os.path.join(ROOT, f)), f


def test_install_hints_name_an_extra_that_installs_the_module():
    from bantamtrain.cli import doctor
    extras = {e: {re.split(r"[<>=\[ ]", r)[0] for r in reqs}
              for e, reqs in _toml()["project"]["optional-dependencies"].items()}
    for mod, extra in cli.EXTRAS.items():
        if extra is not None:
            assert mod in extras[extra], (mod, extra)
    for extra, mods in doctor.EXTRAS.items():
        assert set(mods) <= extras[extra], extra


def test_doctor_report(capsys):
    from bantamtrain.cli import doctor
    assert cli.main(["doctor", "--json"]) == 0
    rep = json.loads(capsys.readouterr().out)
    for key in ("bantamtrain", "python", "packages", "torch", "cuda", "mps", "flush_denormal",
                "threads", "ram", "disk", "extras"):
        assert key in rep
    assert rep["bantamtrain"] == bantamtrain.__version__
    assert rep["packages"]["torch"] and rep["threads"]["torch_intraop"] >= 1
    assert set(rep["extras"]) == {"data", "finetune", "weather", "test"}
    import torch
    state = float(torch.tensor([1e-40]) * 2) == 0.0
    assert rep["flush_denormal"]["enabled"] == state                  # read-only probe
    assert (float(torch.tensor([1e-40]) * 2) == 0.0) == state
    text = doctor.format_report(rep)
    assert "flush-denormal" in text and "threads" in text and "extra finetune" in text


def test_doctor_survives_odd_platforms(monkeypatch):
    """No psutil, no /proc, no os.sysconf (Windows), a stub module without __spec__."""
    from bantamtrain.cli import doctor

    def no_proc(path, *a, **kw):
        raise OSError(path)

    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(doctor, "open", no_proc, raising=False)
    monkeypatch.delattr(os, "sysconf")
    monkeypatch.setitem(sys.modules, "zarr", types.ModuleType("zarr"))
    rep = doctor.collect()
    assert rep["ram"] == {"available_bytes": None, "total_bytes": None}
    assert rep["extras"]["weather"]["zarr"] is False
    assert "ram         ? available of ?" in doctor.format_report(rep)


@pytest.mark.slow
def test_release_script_builds_with_readme_and_license(tmp_path):
    pytest.importorskip("build")
    pytest.importorskip("twine")
    env = {**ENV, "BUILD_DIR": str(tmp_path), "BUILD_ARGS": "--no-isolation"}
    p = subprocess.run(["bash", os.path.join(ROOT, "packaging", "release.sh")], env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    files = sorted(os.listdir(tmp_path / "dist"))
    assert len(files) == 2 and files[0].endswith(".whl") and files[1].endswith(".tar.gz")
    assert bantamtrain.__version__ in files[0]
    assert "README and LICENSE present" in p.stdout
    assert not os.path.exists(os.path.join(ROOT, "bantamtrain.egg-info"))


def test_markers_registered(request):
    names = {m.split(":")[0].strip() for m in request.config.getini("markers")}
    assert {"slow", "network", "gpu", "dist"} <= names


def test_ci_workflow():
    yaml = pytest.importorskip("yaml")
    path = os.path.join(os.path.dirname(ROOT), ".github", "workflows", "ci.yml")
    with open(path) as f:
        wf = yaml.safe_load(f)
    test, pkg = wf["jobs"]["test"], wf["jobs"]["package"]
    assert test["strategy"]["matrix"]["python"] == ["3.10", "3.13"]
    steps = " ".join(s.get("run", "") for s in test["steps"])
    assert 'pip install -e "bantam[test,finetune,data]"' in steps
    assert '-m "not slow and not network and not gpu"' in steps
    steps = " ".join(s.get("run", "") for s in pkg["steps"])
    assert "packaging/release.sh" in steps and "twine" in steps
