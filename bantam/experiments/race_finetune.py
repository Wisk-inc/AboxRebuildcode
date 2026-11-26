"""Fine-tuning race: Hugging Face Trainer + PEFT against `bantam finetune` on the same data,
model and budget, measuring speed, memory and quality.

    python experiments/race_finetune.py --threads 4                                  # full race
    python experiments/race_finetune.py --max_examples 32 --judge_n 16 --out results/race/smoke

Protocol (every arm): HuggingFaceTB/SmolLM2-135M in fp32 on CPU; Dolly-15k from the HF cache,
row indices shuffled with Random(1234): the first 800 train, the next 200 are the held-out
judge set (prompt = instruction, a blank line and the context; both sides stripped); ChatML;
loss on the response tokens only; LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/
down_proj; LR 2e-4, 3% linear warmup then cosine to 10% (Bantam's lr_at), AdamW betas
(0.9, 0.95), eps 1e-8, no weight decay, grad-norm clip 1.0; seq 512, micro-batch 2 x grad
accumulation 2, 1 epoch, seed 1234.

Arms (each trains in its own subprocess; the RSS of its process tree is sampled every 0.2 s)
  base          no training: the judge's reference
  hf_peft       transformers Trainer + peft LoraConfig: one example per row, dynamic padding,
                torch AdamW (optim adamw_torch, scheduler cosine_with_min_lr)
  bantam_adamw  bantam finetune --optim adamw --state_bits 32 (torch AdamW) with packing:
                the same optimizer as hf_peft, so the difference is the training loop
  bantam        Bantam's recipe: --optim muon (Muon on the LoRA matrices), 8-bit optimizer
                state, best-fit-decreasing packing with block-diagonal masks, chunked CE
  bantam_nf4    the same on an NF4 base (QLoRA) built by qweights.load_quantized
                (--quant nf4 --low_ram 1, embeddings int8)

The judge (its own subprocess per arm) scores every final model the same way: per-token CE
over the response tokens of the held-out examples, one example at a time, no packing, fp32
logits from the model's own forward, pooled over tokens; plus greedy answers to fixed
prompts. Finished phases are skipped on a rerun (train / judge hashes); --force reruns arms.
Output: <out>/<arm>.json, <out>/summary.json, <out>/summary.md; adapters, logs and the data
split in <out>/runs and <out>/data (git-ignored).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MODEL = "HuggingFaceTB/SmolLM2-135M"
DATASET = "databricks/databricks-dolly-15k"
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# fixed inside bantam finetune (build_optimizer's betas, lr_at's final=0.1, rt.step's clip);
# the HF arm is set to the same values
MATCHED = {"betas": [0.9, 0.95], "eps": 1e-8, "weight_decay": 0.0, "min_lr_rate": 0.1, "max_grad_norm": 1.0}
# = bantamtrain.finetune.CHATML (tested); not imported, since importing bantamtrain.finetune
# turns on flush-to-zero, which the HF arm must not get by accident
CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + "
          "message['content'] + '<|im_end|>' + '\n' }}{% endfor %}"
          "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
GEN_PROMPTS = ("What is the capital of France?",
               "Give me three tips for staying healthy.",
               ("Summarize in one sentence: The Great Wall of China is a series of fortifications "
                "built across the historical northern borders of ancient Chinese states to protect "
                "them against nomadic groups from the Eurasian Steppe."))
ARMS = {
    "base": {"kind": "base", "desc": "base model, no training (reference)"},
    "hf_peft": {"kind": "hf", "desc": "transformers Trainer + peft, dynamic padding, torch AdamW"},
    "bantam_adamw": {"kind": "bantam", "desc": "bantam finetune, torch AdamW (fp32 state), packing",
                     "flags": ["--optim", "adamw", "--state_bits", "32", "--pack", "bfd"]},
    "bantam": {"kind": "bantam", "desc": "bantam finetune defaults: Muon LoRA, 8-bit state, packing, chunked CE",
               "flags": ["--optim", "muon", "--pack", "bfd"]},
    "bantam_nf4": {"kind": "bantam", "desc": "bantam defaults on an NF4 base (QLoRA via load_quantized)",
                   "flags": ["--optim", "muon", "--pack", "bfd", "--quant", "nf4", "--low_ram", "1"],
                   "quant": "nf4", "embed_bits": 8},
    # follow-ups: packing puts ~3.4 examples in a 512-token window, so at the protocol's 2x2
    # windows per step Bantam takes 58 optimizer steps where hf_peft takes 200. Micro-batch 1 x
    # accum 1 gives 235 steps of ~3.4 examples (hf_peft: 200 steps of 4 examples).
    "bantam_adamw_steps": {"kind": "bantam", "desc": "bantam_adamw with 1 window per optimizer step "
                           "(235 steps, close to hf_peft's 200)",
                           "flags": ["--optim", "adamw", "--state_bits", "32", "--pack", "bfd"],
                           "override": {"batch": 1, "accum": 1}},
    "bantam_muon_steps": {"kind": "bantam", "desc": "Muon LoRA (8-bit state) with 1 window per step",
                          "flags": ["--optim", "muon", "--pack", "bfd"], "override": {"batch": 1, "accum": 1}},
    "bantam_muon_steps_lr1e-3": {"kind": "bantam", "desc": "Muon LoRA with 1 window per step, LR 1e-3",
                                 "flags": ["--optim", "muon", "--pack", "bfd"],
                                 "override": {"batch": 1, "accum": 1, "lr": 1e-3}},
    "bantam_nf4_adamw_steps": {"kind": "bantam", "desc": "QLoRA NF4 base, torch AdamW, 1 window per step",
                               "flags": ["--optim", "adamw", "--state_bits", "32", "--pack", "bfd",
                                         "--quant", "nf4", "--low_ram", "1"],
                               "quant": "nf4", "embed_bits": 8, "override": {"batch": 1, "accum": 1}},
}
DEFAULT_ARMS = "base,hf_peft,bantam_adamw,bantam,bantam_nf4"

try:
    import psutil
except ImportError:                                  # /proc fallback below
    psutil = None


# ----------------------------------------------------------------------------- small helpers
def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- data
def to_example(row):
    """A Dolly row (instruction / context / response) or a prompt / response row ->
    {prompt, response}, both stripped; the context follows the instruction after a blank
    line (as bantam finetune joins Dolly columns). None when a side is empty."""
    prompt = str(row.get("instruction") or row.get("prompt") or "").strip()
    ctx = str(row.get("context") or "").strip()
    resp = str(row.get("response") or "").strip()
    if ctx and prompt:
        prompt = f"{prompt}\n\n{ctx}"
    return {"prompt": prompt, "response": resp} if prompt and resp else None


def split_rows(rows, n_train, n_judge, seed, usable=None, stats=None):
    """Deterministic disjoint split: row indices shuffled with Random(seed); the first n_train
    usable rows train, the next n_judge are judged (usable: to_example gives an example and
    usable(example), e.g. a response token is left within seq). Examples keep their row index
    as id; `stats` (a dict) gets the number of unusable rows passed over."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    train, judge, bad = [], [], 0
    for i in idx:
        ex = to_example(rows[i])
        if ex is None or (usable is not None and not usable(ex)):
            bad += 1
            continue
        (train if len(train) < n_train else judge).append({"id": i, **ex})
        if len(judge) == n_judge:
            break
    if stats is not None:
        stats["unusable"] = bad
    if len(train) < n_train or len(judge) < n_judge:
        raise SystemExit(f"[race] {len(rows)} rows give {len(train)} train / {len(judge)} judge "
                         f"examples; need {n_train} / {n_judge}")
    return train, judge


def as_messages(ex):
    return {"id": ex["id"], "messages": [{"role": "user", "content": ex["prompt"]},
                                         {"role": "assistant", "content": ex["response"]}]}


def from_messages(row):
    m = row["messages"]
    return {"id": row.get("id"), "prompt": m[0]["content"], "response": m[1]["content"]}


def race_tokenizer(model):
    """The tokenizer as every arm uses it, set up as bantam finetune does: the model's chat
    template, else ChatML (the vocab has its tokens) with eos = <|im_end|>; pad = eos."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    if not getattr(tok, "chat_template", None):
        if "<|im_end|>" not in tok.get_vocab():
            raise SystemExit(f"[race] {model}: no chat template and no ChatML tokens")
        tok.chat_template = CHATML
        tok.eos_token = "<|im_end|>"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def encode(tok, ex, seq, stats=None):
    """-> (ids, labels), the way a Trainer user builds SFT data: the prompt is the chat
    template with the generation prompt, labels are -100 on it, the sequence is cut to seq
    tokens. None when no response token is left after the cut."""
    user = [{"role": "user", "content": ex["prompt"]}]
    p = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(user + [{"role": "assistant", "content": ex["response"]}], tokenize=False)
    p_ids = tok(p, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    n = len(p_ids)
    if ids[:n] != p_ids:                      # the prompt is not a token prefix: longest common one
        n = next((i for i, (a, b) in enumerate(zip(ids, p_ids)) if a != b), min(len(ids), n))
        if stats is not None:
            stats["boundary_fallback"] = stats.get("boundary_fallback", 0) + 1
    labels = [-100] * n + ids[n:]
    if stats is not None and len(ids) > seq:
        stats["truncated"] = stats.get("truncated", 0) + 1
    ids, labels = ids[:seq], labels[:seq]
    if all(t == -100 for t in labels[1:]):
        if stats is not None:
            stats["no_target"] = stats.get("no_target", 0) + 1
        return None
    return ids, labels


def bantam_args(jsonl, seed):
    return argparse.Namespace(jsonl=jsonl, dataset="", config="", split="train", max_examples=0, seed=seed,
                              messages_col="messages", prompt_col="prompt", response_col="response")


def token_parity(tok, jsonl, seq):
    """Does encode() give exactly the ids and labels bantam finetune trains on (its own
    load_examples + tokenize_examples on the same JSONL)?"""
    from bantamtrain import finetune as ft
    mine = [e for e in (encode(tok, from_messages(r), seq) for r in read_jsonl(jsonl)) if e is not None]
    theirs = ft.tokenize_examples(tok, ft.load_examples(bantam_args(jsonl, 0)), seq)
    bad = sum(1 for (i, l), (i2, l2, _) in zip(mine, theirs) if i != i2 or l != l2)
    return {"examples": len(mine), "bantam_examples": len(theirs),
            "mismatched": bad + abs(len(mine) - len(theirs)), "ok": bad == 0 and len(mine) == len(theirs)}


def bantam_plan(tok, jsonl, p, pack="bfd"):
    """Replay bantam finetune's data plan (tokenize, shuffle, BFD pack, plan_steps, window
    order) with its own functions -> the windows it trains on, the real (non-pad) tokens and
    supervised targets they hold, and the targets of every step (checked against log.json)."""
    from bantamtrain import finetune as ft
    items = ft.tokenize_examples(tok, ft.load_examples(bantam_args(jsonl, p["seed"])), p["seq"], 0.0, True)
    random.Random(p["seed"]).shuffle(items)
    win = ft.pack(ft.with_prompt_loss(items, 0.0), p["seq"], tok.pad_token_id, pack)
    per, steps = ft.plan_steps(len(win), 1, p["batch"], p["accum"], p["epochs"], 0)
    rng = random.Random(p["seed"])
    order = list(range(per))
    rng.shuffle(order)
    pos, used, step_targets = 0, [], []
    for _ in range(steps):
        n = 0
        for _ in range(p["accum"]):
            if pos + p["batch"] > len(order):
                rng.shuffle(order)
                pos = 0
            for i in order[pos:pos + p["batch"]]:
                used.append(i)
                n += sum(1 for t in win[i][1][1:] if t != -100)
            pos += p["batch"]
        step_targets.append(n)
    sup = [sum(1 for t in w[1][1:] if t != -100) for w in win]
    return {"windows": len(win), "steps": steps, "windows_used": len(used),
            "real_tokens": sum(len(win[i][3]) - win[i][3].count(-1) for i in used),
            "supervised_tokens": sum(sup[i] for i in used), "supervised_tokens_all": sum(sup),
            "step_targets": step_targets}


def check_plan(plan, log):
    """The replayed plan against bantam's log.json: window, step and target counts."""
    logged = {e["step"]: e.get("targets") for e in log.get("train", [])}
    steps_ok = all(t is None or round(t) == plan["step_targets"][s - 1] for s, t in logged.items()
                   if 0 < s <= len(plan["step_targets"]))
    return (plan["windows"] == log.get("train_windows") and plan["steps"] == log.get("steps")
            and plan["supervised_tokens_all"] == log.get("supervised_tokens") and steps_ok)


# ----------------------------------------------------------------------------- measurement
def _proc_children(pid):
    out = []
    try:
        for tid in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{tid}/children") as f:
                out += [int(c) for c in f.read().split()]
    except (OSError, ValueError):
        pass
    return out


def _proc_rss(pid):
    try:
        with open(f"/proc/{pid}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def tree_rss(pid):
    """Bytes resident in pid and all its descendants (psutil when installed, else /proc)."""
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            procs = [p] + p.children(recursive=True)
        except psutil.Error:
            return 0
        tot = 0
        for q in procs:
            try:
                tot += q.memory_info().rss
            except psutil.Error:
                pass
        return tot
    seen, stack, tot = set(), [pid], 0
    while stack:
        q = stack.pop()
        if q not in seen:
            seen.add(q)
            tot += _proc_rss(q)
            stack += _proc_children(q)
    return tot


class RssSampler(threading.Thread):
    """Samples tree_rss(pid) every `interval` seconds until stop(); keeps the peak and a trace."""

    def __init__(self, pid, interval=0.2):
        super().__init__(daemon=True)
        self.pid, self.interval, self.peak, self.trace = pid, interval, 0, []
        self._halt = threading.Event()
        self._t0 = time.perf_counter()

    def run(self):
        while True:
            r = tree_rss(self.pid)
            self.peak = max(self.peak, r)
            self.trace.append((round(time.perf_counter() - self._t0, 1), r))
            if self._halt.wait(self.interval):
                return

    def stop(self):
        self._halt.set()
        self.join()


def mem_available_gb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None


def run_measured(cmd, log_path, env=None, cwd=None, tag="", interval=0.2, echo=True):
    """Run cmd with its output teed to log_path while a thread samples the RSS of its process
    tree. -> returncode, wall_s, cpu_s (user + sys of the process and its children, from
    wait4), maxrss_mb (wait4), peak_rss_mb (sampled), a trace of at most ~300 samples."""
    load0, mem0 = os.getloadavg(), mem_available_gb()
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd,
                             text=True, bufsize=1)
        sampler = RssSampler(p.pid, interval)
        sampler.start()
        for line in p.stdout:
            log.write(line)
            log.flush()
            if echo:
                print(f"  [{tag}] {line.rstrip()}", flush=True)
        p.stdout.close()
        _, status, ru = os.wait4(p.pid, 0)
        sampler.stop()
        p.returncode = os.waitstatus_to_exitcode(status)
    wall = time.perf_counter() - t0
    k = max(1, len(sampler.trace) // 300)
    return {"returncode": p.returncode, "wall_s": wall, "cpu_s": ru.ru_utime + ru.ru_stime,
            "maxrss_mb": ru.ru_maxrss / 1024, "peak_rss_mb": sampler.peak / 2**20, "samples": len(sampler.trace),
            "rss_trace_mb": [[t, round(r / 2**20, 1)] for t, r in sampler.trace[::k]],
            "loadavg_before": list(load0), "loadavg_after": list(os.getloadavg()),
            "mem_available_gb_before": mem0}


def tail(path, n=30):
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


# ----------------------------------------------------------------------------- environment
def versions():
    from importlib import metadata

    import bantamtrain
    v = {}
    for pkg in ("torch", "transformers", "peft", "accelerate", "datasets", "tokenizers", "safetensors",
                "numpy", "psutil"):
        try:
            v[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            v[pkg] = None
    v["bantamtrain"] = bantamtrain.__version__
    v["python"] = platform.python_version()
    try:
        git = ["git", "-C", ROOT]
        kw = {"capture_output": True, "text": True, "timeout": 10, "check": False}
        head = subprocess.run(git + ["rev-parse", "HEAD"], **kw).stdout.strip()
        dirty = subprocess.run(git + ["status", "--porcelain", "--untracked-files=no"], **kw).stdout.strip()
        v["bantam_git"] = head + ("-dirty" if dirty else "") if head else None
    except (OSError, subprocess.SubprocessError):
        v["bantam_git"] = None
    return v


def machine():
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next((line.split(":", 1)[1].strip() for line in f if line.startswith("model name")), None)
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            mem = int(f.readline().split()[1]) / 2**20
    except (OSError, ValueError, IndexError):
        mem = None
    aff = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"platform": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(), "usable_cpus": aff,
            "mem_total_gb": mem, "loadavg": list(os.getloadavg())}


def child_env(threads, online=False):
    env = dict(os.environ)
    path = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", PYTHONPATH=path)
    if not online:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


# ----------------------------------------------------------------------------- the judge
@torch.no_grad()
def judge_scores(model, items):
    """The shared judge: [(CE sum, targets)] per example over its response tokens; one
    example at a time (no padding, no packing), fp32 logits from the model's own forward."""
    model.eval()
    dev = next(model.parameters()).device
    out = []
    for ids, labels in items:
        logits = model(input_ids=torch.tensor([ids], device=dev)).logits[0, :-1].float()
        y = torch.tensor(labels[1:], device=dev)
        out.append((F.cross_entropy(logits, y, ignore_index=-100, reduction="sum").item(),
                    int((y != -100).sum())))
    return out


def pooled(scores):
    s, n = sum(a for a, _ in scores), sum(b for _, b in scores)
    return s / n if n else float("nan")


def bootstrap_ci(a, b=None, n=1000, seed=0):
    """95% interval of pooled(a) (or pooled(a) - pooled(b), paired by example) over example resamples."""
    r = random.Random(seed)
    k = len(a)
    vals = []
    for _ in range(n):
        idx = [r.randrange(k) for _ in range(k)]
        v = pooled([a[i] for i in idx])
        vals.append(v - pooled([b[i] for i in idx]) if b is not None else v)
    vals.sort()
    return [vals[int(0.025 * n)], vals[int(0.975 * n) - 1]]


@torch.no_grad()
def greedy_answers(model, tok, prompts, max_new):
    dev = next(model.parameters()).device
    eos = [tok.eos_token_id]
    g = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    eos += [i for i in (g if isinstance(g, list) else [g]) if i is not None and i not in eos]
    out = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                       add_generation_prompt=True)
        x = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]], device=dev)
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=max_new, do_sample=False,
                           temperature=None, top_p=None, top_k=None, eos_token_id=eos,
                           pad_token_id=tok.pad_token_id)[0, x.shape[1]:]
        out.append({"prompt": p, "answer": tok.decode(y, skip_special_tokens=True).strip(), "new_tokens": len(y)})
    return out


def load_final(p, spec, run, float_base=False):
    """The arm's final model in fp32: the float base + the PEFT adapter (hf_peft's own, or the
    one bantam finetune exports), or for a quantized arm the base it trained on
    (qweights.load_quantized) + its adapter.pt. float_base: a quantized arm's adapter on the
    float base instead (what --save_merged gives)."""
    from transformers import AutoModelForCausalLM
    if spec.get("quant") and not float_base:
        from bantamtrain.cli.chat import attach_adapter
        from bantamtrain.qweights import load_quantized
        res = load_quantized(p["model"], spec["quant"], embed_bits=spec.get("embed_bits", 8), dtype=torch.float32)
        model = res[0] if isinstance(res, tuple) else res
        attach_adapter(model, run)
        return model.eval(), f"qweights.load_quantized({spec['quant']}) + adapter.pt"
    model = AutoModelForCausalLM.from_pretrained(p["model"], dtype=torch.float32)
    if spec["kind"] == "base":
        return model.eval(), "base model"
    from peft import PeftModel
    path = os.path.join(run, "adapter") if spec["kind"] == "hf" else run
    return PeftModel.from_pretrained(model, path).eval(), "float base + PeftModel.from_pretrained"


def judge_worker(out, arm):
    cfg = read_json(os.path.join(out, "race_config.json"))
    p, spec, run = cfg["protocol"], cfg["arms"][arm], os.path.join(out, "runs", arm)
    torch.set_num_threads(p["threads"])
    tok = race_tokenizer(p["model"])
    rows = read_jsonl(cfg["files"]["judge"])
    n = max(cfg["judge"]["judge_n"], p["bantam_eval_n"])
    enc = [(r["id"], encode(tok, from_messages(r), p["seq"])) for r in rows[:n]]
    jn = cfg["judge"]["judge_n"]
    res = {"how": None, "judge_n": jn, "skipped": sum(1 for _, e in enc[:jn] if e is None)}

    def score(model):
        t0 = time.perf_counter()
        sc = judge_scores(model, [e for _, e in enc if e is not None])
        ids = [i for i, e in enc if e is not None]
        k_j = sum(1 for _, e in enc[:jn] if e is not None)
        k_s = sum(1 for _, e in enc[:p["bantam_eval_n"]] if e is not None)
        j = sc[:k_j]
        return {"judge_loss": pooled(j), "judge_ppl": math.exp(pooled(j)), "judge_ci95": bootstrap_ci(j),
                "judge_macro_loss": sum(a / b for a, b in j) / max(1, len(j)), "judge_examples": len(j),
                "judge_tokens": sum(b for _, b in j), "subset_loss": pooled(sc[:k_s]), "subset_examples": k_s,
                "scores": [[i, a, b] for i, (a, b) in zip(ids[:k_j], j)], "scored_examples": len(sc),
                "score_s": time.perf_counter() - t0}

    t0 = time.perf_counter()
    model, res["how"] = load_final(p, spec, run)
    res["load_s"] = time.perf_counter() - t0
    res.update(score(model))
    t0 = time.perf_counter()
    res["generations"] = greedy_answers(model, tok, cfg["judge"]["prompts"], cfg["judge"]["gen_tokens"])
    res["generate_s"] = time.perf_counter() - t0
    if spec.get("quant") and cfg["judge"]["float_base"]:
        del model
        model, how = load_final(p, spec, run, float_base=True)
        fb = score(model)
        res["float_base"] = {k: fb[k] for k in ("judge_loss", "judge_ppl", "judge_ci95", "subset_loss")}
        res["float_base"]["how"] = how
    write_json(os.path.join(run, "judge_inner.json"), res)


# ----------------------------------------------------------------------------- the HF arm
def hf_worker(out, arm):
    """transformers Trainer + peft on the shared data: one example per row, dynamic padding."""
    cfg = read_json(os.path.join(out, "race_config.json"))
    p, run = cfg["protocol"], os.path.join(out, "runs", arm)
    torch.set_num_threads(p["threads"])
    if p.get("hf_flush_denormal"):                  # bantam finetune always does this at import
        torch.set_flush_denormal(True)
    import peft
    from transformers import (AutoModelForCausalLM, DataCollatorForSeq2Seq, Trainer, TrainerCallback,
                              TrainingArguments, set_seed)
    t_start = time.perf_counter()
    tok = race_tokenizer(p["model"])
    items = [e for e in (encode(tok, from_messages(r), p["seq"]) for r in read_jsonl(cfg["files"]["train"]))
             if e is not None]
    data = [{"input_ids": i, "attention_mask": [1] * len(i), "labels": l} for i, l in items]
    set_seed(p["seed"])
    model = AutoModelForCausalLM.from_pretrained(p["model"], dtype=torch.float32)
    model = peft.get_peft_model(model, peft.LoraConfig(
        r=p["lora_r"], lora_alpha=p["lora_alpha"], lora_dropout=p["lora_dropout"],
        target_modules=list(p["targets"]), bias="none", task_type="CAUSAL_LM"))
    lora_mods = [n for n, m in model.named_modules()
                 if isinstance(getattr(m, "lora_A", None), torch.nn.ModuleDict)]

    class Counting:
        """The collator, counting what training sees: non-pad tokens, positions, targets."""

        def __init__(self, inner):
            self.inner, self.tokens, self.positions, self.targets = inner, 0, 0, 0

        def __call__(self, feats):
            b = self.inner(feats)
            self.tokens += int(b["attention_mask"].sum())
            self.positions += b["attention_mask"].numel()
            self.targets += int((b["labels"][:, 1:] != -100).sum())
            return b

    class Clock(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            self.t0, self.c0 = time.perf_counter(), time.process_time()

        def on_train_end(self, args, state, control, **kw):
            self.wall, self.cpu = time.perf_counter() - self.t0, time.process_time() - self.c0

    collate = Counting(DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100, return_tensors="pt"))
    clock = Clock()
    targs = TrainingArguments(
        output_dir=os.path.join(run, "trainer"), per_device_train_batch_size=p["batch"],
        gradient_accumulation_steps=p["accum"], num_train_epochs=p["epochs"], learning_rate=p["lr"],
        lr_scheduler_type="cosine_with_min_lr", lr_scheduler_kwargs={"min_lr_rate": MATCHED["min_lr_rate"]},
        warmup_steps=p["warmup"], weight_decay=MATCHED["weight_decay"], adam_beta1=MATCHED["betas"][0],
        adam_beta2=MATCHED["betas"][1], adam_epsilon=MATCHED["eps"], max_grad_norm=MATCHED["max_grad_norm"],
        optim="adamw_torch", logging_steps=1, save_strategy="no", eval_strategy="no", report_to="none",
        use_cpu=True, seed=p["seed"], data_seed=p["seed"], dataloader_num_workers=0, dataloader_pin_memory=False,
        disable_tqdm=True, remove_unused_columns=False, label_names=["labels"], bf16=False, fp16=False)
    trainer = Trainer(model=model, args=targs, train_dataset=data, data_collator=collate, callbacks=[clock])
    load_s = time.perf_counter() - t_start
    tr = trainer.train()
    hist = [[e["step"], e["loss"], e.get("learning_rate")] for e in trainer.state.log_history if "loss" in e]
    model.save_pretrained(os.path.join(run, "adapter"))
    tok.save_pretrained(os.path.join(run, "adapter"))
    steps = trainer.state.global_step
    write_json(os.path.join(run, "hf_train.json"), {
        "train_wall_s": clock.wall, "train_cpu_s": clock.cpu, "load_s": load_s, "steps": steps,
        "examples": len(data), "real_tokens": collate.tokens, "supervised_tokens": collate.targets,
        "padding_fraction": 1 - collate.tokens / max(1, collate.positions), "losses": hist,
        "mean_train_loss": tr.training_loss, "trainable_params": model.get_nb_trainable_parameters()[0],
        "lora_modules": len(lora_mods), "lora_leaves": sorted({n.split(".")[-1] for n in lora_mods}),
        "warmup_steps": targs.get_warmup_steps(steps),
        "settings": {"optimizer": "torch.optim.AdamW (optim=adamw_torch), fp32 state",
                     "schedule": f"cosine_with_min_lr (min_lr_rate {MATCHED['min_lr_rate']}), "
                                 f"warmup_steps={p['warmup']}",
                     "batching": "one example per row, dynamic padding (DataCollatorForSeq2Seq)",
                     "loss": "model forward with labels (full logits), tokens of the whole step averaged",
                     "base": "fp32", "betas": MATCHED["betas"],
                     "attn_implementation": getattr(model.config, "_attn_implementation", None),
                     "loss_over_step_tokens": bool(getattr(trainer, "model_accepts_loss_kwargs", False)),
                     "flush_denormal": bool(p.get("hf_flush_denormal"))}})


# ----------------------------------------------------------------------------- orchestration
def bantam_cmd(cfg, spec, run):
    p, f = {**cfg["protocol"], **spec.get("override", {})}, cfg["files"]
    return [sys.executable, "-m", "bantamtrain", "finetune", "--model", p["model"], "--jsonl", f["train"],
            "--eval_jsonl", f["bantam_eval"], "--seq", str(p["seq"]), "--batch", str(p["batch"]),
            "--accum", str(p["accum"]), "--epochs", str(p["epochs"]), "--lr", str(p["lr"]), "--sched", "cosine",
            "--warmup", str(p["warmup"]), "--wd", "0", "--lora", str(p["lora_r"]),
            "--lora_alpha", str(p["lora_alpha"]), "--lora_dropout", str(p["lora_dropout"]),
            "--lora_targets", ",".join(p["targets"]), "--seed", str(p["seed"]), "--threads", str(p["threads"]),
            "--eval_every", "0", "--save_peft", "1", *spec.get("flags", []), "--out", run]


def worker_cmd(what, out, arm):
    return [sys.executable, os.path.abspath(__file__), what, "--out", out, "--arm", arm]


def tail_loss(losses, steps):
    t = [l for s, l, *_ in losses if s > 0.75 * steps] or [l for _, l, *_ in losses[-1:]]
    return sum(t) / len(t) if t else None


def bantam_train_result(run, plan):
    log = read_json(os.path.join(run, "log.json")) or {}
    a = log.get("args", {})
    losses = [[e["step"], e["loss"], e.get("lr")] for e in log.get("train", [])]
    lora = log.get("report", {}).get("lora", {})
    return {"train_wall_s": log.get("train_time"), "train_cpu_s": log.get("train_cpu_time"),
            "steps": log.get("steps"), "examples": log.get("examples"), "real_tokens": plan["real_tokens"],
            "supervised_tokens": plan["supervised_tokens"], "supervised_tokens_all": log.get("supervised_tokens"),
            "windows": log.get("train_windows"), "windows_used": plan["windows_used"],
            "plan_matches_log": check_plan(plan, log), "padding_fraction": log.get("padding_fraction"),
            "losses": losses, "trainable_params": log.get("trainable_params"),
            "lora_modules": lora.get("wrapped"), "lora_leaves": lora.get("targets"),
            "eval_loss_step0": (log.get("evals") or [{}])[0].get("eval_loss"),
            "final_eval_loss": log.get("final_eval_loss"),
            "settings": {"optimizers": log.get("optimizers"), "optim": a.get("optim"),
                         "muon_lora": a.get("muon_lora"), "state_bits": a.get("state_bits"), "pack": a.get("pack"),
                         "chunked_ce": log.get("chunked_ce"), "ce_chunk": a.get("ce_chunk"),
                         "quant": a.get("quant"),
                         "embed_quant": a.get("embed_quant") if a.get("quant") not in (None, "0") else None,
                         "low_ram": a.get("low_ram"), "sched": a.get("sched"), "warmup": a.get("warmup"),
                         "param_dtype": a.get("param_dtype"), "precision": a.get("precision"),
                         "model_bytes": log.get("model_bytes"),
                         "optimizer_state_bytes": log.get("optimizer_state_bytes"),
                         "data": log.get("data"), "flush_denormal": True, "betas": MATCHED["betas"]}}


def arm_hashes(cfg, name):
    """(train hash, judge hash): training depends on the protocol, the arm and the data files;
    the judge on the trained model and the judge settings."""
    arm = {k: v for k, v in cfg["arms"][name].items() if k != "desc"}
    th = digest({"protocol": cfg["protocol"], "arm": arm, "files": cfg["sha256"]})
    return th, digest({"train": th, "judge": cfg["judge"]})


def run_arm(name, cfg, args, state):
    out, p = args.out, cfg["protocol"]
    spec = cfg["arms"][name]
    run = os.path.join(out, "runs", name)
    th, jh = arm_hashes(cfg, name)
    force = name in args.force or "all" in args.force
    rec_path = os.path.join(out, f"{name}.json")
    old = read_json(rec_path)
    if old and old.get("status") == "done" and old.get("train_hash") == th and old.get("judge_hash") == jh \
            and not force:
        print(f"[race] {name}: done, skipped (--force {name} reruns it)", flush=True)
        return old
    env = child_env(p["threads"], args.online)
    rec = {"arm": name, "kind": spec["kind"], "desc": spec["desc"], "status": "running", "train_hash": th,
           "judge_hash": jh, "protocol": p, "versions": state["versions"], "machine": machine(), "commands": {}}
    train, retrained = None, False
    if spec["kind"] != "base":
        prev = read_json(os.path.join(run, "train_result.json"))
        if prev and prev.get("train_hash") == th and not force:
            train = prev
            print(f"[race] {name}: training done earlier, judging only", flush=True)
        else:
            if os.path.isdir(run):
                shutil.rmtree(run)
            os.makedirs(run)
            cmd = bantam_cmd(cfg, spec, run) if spec["kind"] == "bantam" else worker_cmd("_hf", out, name)
            rec["commands"]["train"] = shlex.join(cmd)
            print(f"[race] {name}: training\n  $ {shlex.join(cmd)}", flush=True)
            m = run_measured(cmd, os.path.join(run, "train.log"), env, ROOT, name, echo=not args.quiet)
            if m["returncode"]:
                rec.update(status="failed", error=f"training exited {m['returncode']}", log_tail=tail(
                    os.path.join(run, "train.log")), train={"process": m})
                write_json(rec_path, rec)
                print(f"[race] {name}: training FAILED (see {run}/train.log)", flush=True)
                return rec
            if spec["kind"] == "bantam":
                fl = spec.get("flags", [])
                pack = fl[fl.index("--pack") + 1] if "--pack" in fl else "bfd"
                pa = {**p, **spec.get("override", {})}
                key = (pack, pa["batch"], pa["accum"], pa["epochs"])
                if key not in state.setdefault("plans", {}):
                    state["plans"][key] = bantam_plan(state["tok"], cfg["files"]["train"], pa, pack)
                train = bantam_train_result(run, state["plans"][key])
            else:
                train = read_json(os.path.join(run, "hf_train.json"))
            if not train or train.get("train_wall_s") is None:
                rec.update(status="failed", error="training wrote no result", train={"process": m})
                write_json(rec_path, rec)
                print(f"[race] {name}: training wrote no result (see {run})", flush=True)
                return rec
            train.update(train_hash=th, command=shlex.join(cmd), process=m)
            retrained = True
            wall = train.get("train_wall_s") or 0
            train["tokens_per_s"] = train["real_tokens"] / wall if wall else None
            train["supervised_tokens_per_s"] = train["supervised_tokens"] / wall if wall else None
            train["final_train_loss"] = train["losses"][-1][1] if train["losses"] else None
            train["train_loss_tail"] = tail_loss(train["losses"], train["steps"] or 1)
            write_json(os.path.join(run, "train_result.json"), train)
        rec["commands"]["train"] = train["command"]
    os.makedirs(run, exist_ok=True)
    judge = read_json(os.path.join(run, "judge.json"))
    if force or retrained or not judge or judge.get("judge_hash") != jh:
        cmd = worker_cmd("_judge", out, name)
        print(f"[race] {name}: judging\n  $ {shlex.join(cmd)}", flush=True)
        m = run_measured(cmd, os.path.join(run, "judge.log"), env, ROOT, name + ":judge", echo=not args.quiet)
        if m["returncode"]:
            rec.update(status="failed", error=f"judge exited {m['returncode']}",
                       log_tail=tail(os.path.join(run, "judge.log")), train=train)
            write_json(rec_path, rec)
            print(f"[race] {name}: judge FAILED (see {run}/judge.log)", flush=True)
            return rec
        judge = read_json(os.path.join(run, "judge_inner.json"))
        judge.update(judge_hash=jh, command=shlex.join(cmd),
                     process={k: v for k, v in m.items() if k != "rss_trace_mb"})
        write_json(os.path.join(run, "judge.json"), judge)
    rec["commands"]["judge"] = judge["command"]
    rec.update(status="done", train=train, judge=judge)
    write_json(rec_path, rec)
    print(f"[race] {name}: judge loss {fmt(judge['judge_loss'], '.4f')}"
          + (f", {fmt(train['train_wall_s'], '.0f')} s training, {fmt(train['tokens_per_s'], '.0f')} tok/s, "
             f"peak RSS {fmt(train['process']['peak_rss_mb'], '.0f')} MB" if train else ""), flush=True)
    return rec


def prepare(args, state):
    """Split the data, write the JSONL files, check token parity -> race_config.json."""
    p = {"model": args.model, "data": args.data_jsonl or DATASET, "split_seed": args.split_seed,
         "n_train": args.n_train, "n_judge": args.n_judge, "max_examples": args.max_examples,
         "bantam_eval_n": min(args.bantam_eval_n, args.n_judge), "seq": args.seq, "batch": args.batch,
         "accum": args.accum, "epochs": args.epochs, "lr": args.lr, "warmup": args.warmup,
         "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
         "targets": list(TARGETS), "seed": args.seed, "threads": args.threads, "matched": MATCHED,
         "hf_flush_denormal": bool(args.hf_flush_denormal)}
    ddir = os.path.join(args.out, "data")
    os.makedirs(ddir, exist_ok=True)
    if args.data_jsonl:
        rows = read_jsonl(args.data_jsonl)
    else:
        from datasets import load_dataset
        rows = [dict(r) for r in load_dataset(DATASET, split="train")]
    tok = state["tok"] = race_tokenizer(args.model)
    sst = {}
    train, judge = split_rows(rows, args.n_train, args.n_judge, args.split_seed,
                              lambda e: encode(tok, e, args.seq) is not None, sst)
    files = {"train": os.path.join(ddir, "train.jsonl"), "judge": os.path.join(ddir, "judge.jsonl"),
             "bantam_eval": os.path.join(ddir, "bantam_eval.jsonl")}
    write_jsonl(files["train"], [as_messages(e) for e in train[:args.max_examples or None]])
    write_jsonl(files["judge"], [as_messages(e) for e in judge])
    write_jsonl(files["bantam_eval"], [as_messages(e) for e in judge[:p["bantam_eval_n"]]])
    st = {}
    full = [e for e in (encode(tok, e, args.seq, st) for e in train) if e is not None]
    cfg = {"protocol": p, "arms": {k: ARMS[k] for k in ARMS},
           "judge": {"judge_n": min(args.judge_n, args.n_judge), "gen_tokens": args.gen_tokens,
                     "prompts": list(GEN_PROMPTS), "float_base": bool(args.judge_float_base)},
           "files": files, "sha256": {k: sha256(v) for k, v in files.items()},
           "data": {"rows": len(rows), "unusable_rows_skipped": sst["unusable"],
                    "train_ids_head": [e["id"] for e in train[:10]],
                    "judge_ids_head": [e["id"] for e in judge[:10]],
                    "full_train_examples": len(full), "full_train_tokens": sum(len(i) for i, _ in full),
                    "full_train_targets": sum(sum(1 for t in l[1:] if t != -100) for _, l in full),
                    "full_train_encode": st},
           "token_parity": token_parity(tok, files["train"], args.seq)}
    write_json(os.path.join(args.out, "race_config.json"), cfg)
    with open(os.path.join(args.out, ".gitignore"), "w") as f:
        f.write("runs/\ndata/\nrace_config.json\nrace.lock\n")
    if not cfg["token_parity"]["ok"]:
        print(f"[race] WARNING: the HF arm's tokenization differs from bantam finetune's: {cfg['token_parity']}",
              flush=True)
    return cfg


def fmt(v, spec=".3f"):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def summarize(args, cfg, state):
    """summary.json + summary.md over every arm whose record matches the current protocol."""
    recs = {}
    for name in ARMS:
        r = read_json(os.path.join(args.out, f"{name}.json"))
        if r and r.get("status") == "done" and (r.get("train_hash"), r.get("judge_hash")) == arm_hashes(cfg, name):
            recs[name] = r
    base = recs.get("base", {}).get("judge", {}).get("judge_loss")
    ref = recs.get("hf_peft", {}).get("judge", {}).get("scores")
    smoke = bool(cfg["protocol"]["max_examples"])
    arms = {}
    for name, r in recs.items():
        t, j = r.get("train") or {}, r["judge"]
        pr = t.get("process", {})
        a = {"desc": r["desc"], "judge_loss": j["judge_loss"], "judge_ci95": j["judge_ci95"],
             "judge_ppl": j["judge_ppl"], "judge_examples": j["judge_examples"], "judge_tokens": j["judge_tokens"],
             "judge_vs_base": None if base is None else j["judge_loss"] - base,
             "generations": [g["answer"] for g in j["generations"]], "judge_s": j["process"]["wall_s"],
             "judge_peak_rss_mb": j["process"]["peak_rss_mb"]}
        if ref and name != "hf_peft" and [s[0] for s in j["scores"]] == [s[0] for s in ref]:
            a["judge_vs_hf_peft"] = j["judge_loss"] - recs["hf_peft"]["judge"]["judge_loss"]
            a["judge_vs_hf_peft_ci95"] = bootstrap_ci([tuple(s[1:]) for s in j["scores"]],
                                                      [tuple(s[1:]) for s in ref])
        if "float_base" in j:
            a["judge_loss_float_base"] = j["float_base"]["judge_loss"]
        if t:
            a.update({k: t.get(k) for k in ("steps", "examples", "real_tokens", "supervised_tokens",
                                             "tokens_per_s", "supervised_tokens_per_s", "train_wall_s",
                                             "train_cpu_s",
                                             "padding_fraction", "final_train_loss", "train_loss_tail",
                                             "trainable_params", "lora_modules", "settings", "windows",
                                             "windows_used", "plan_matches_log")})
            a.update(peak_rss_mb=pr.get("peak_rss_mb"), maxrss_mb=pr.get("maxrss_mb"),
                     process_wall_s=pr.get("wall_s"), process_cpu_s=pr.get("cpu_s"),
                     loadavg_before=pr.get("loadavg_before"), command=t.get("command"))
            if t.get("final_eval_loss") is not None:           # bantam's own eval on the same examples
                a["bantam_eval_minus_judge_subset"] = t["final_eval_loss"] - j["subset_loss"]
            if smoke and t.get("tokens_per_s"):
                a["est_full_train_min"] = cfg["data"]["full_train_tokens"] / t["tokens_per_s"] / 60
        if smoke:                     # the full judge scores max(n_judge, bantam_eval_n) examples
            n_full = max(cfg["protocol"]["n_judge"], cfg["protocol"]["bantam_eval_n"])
            a["est_full_judge_min"] = (j["load_s"] + j["generate_s"]
                                       + j["score_s"] / max(1, j["scored_examples"]) * n_full) / 60
        arms[name] = a
    trained = {n: a for n, a in arms.items() if "lora_modules" in a}
    sup = {n: a.get("supervised_tokens") for n, a in trained.items()}
    fair = {"token_parity": cfg["token_parity"],
            "same_lora_modules": len({(a["lora_modules"], a["trainable_params"]) for a in trained.values()}) <= 1,
            "supervised_tokens": sup,
            "bantam_plan_matches_log": {n: a.get("plan_matches_log") for n, a in trained.items()
                                        if "windows" in a}}
    if "base" in recs:              # bantam's step-0 eval scores the base model on the same examples
        sub = recs["base"]["judge"]["subset_loss"]
        fair["bantam_step0_eval_minus_base_judge_subset"] = {
            n: r["train"]["eval_loss_step0"] - sub for n, r in recs.items()
            if (r.get("train") or {}).get("eval_loss_step0") is not None and not ARMS[n].get("quant")}
    summary = {"protocol": cfg["protocol"], "judge": cfg["judge"], "data": cfg["data"], "fairness": fair,
               "versions": state["versions"], "machine": machine(), "smoke": smoke, "arms": arms,
               "harness_command": shlex.join([sys.executable] + sys.argv)}
    write_json(os.path.join(args.out, "summary.json"), summary)
    write_markdown(os.path.join(args.out, "summary.md"), summary, cfg)
    return summary


def _min(s):
    return s / 60 if s is not None else None


def write_markdown(path, summary, cfg):
    p, arms, fair, smoke, mach = (summary["protocol"], summary["arms"], summary["fairness"], summary["smoke"],
                                  summary["machine"])
    head = ("| arm | steps | train min | real tok/s | train CPU min | peak RSS GB | final train loss "
            "| judge loss (95% CI) | vs base |")
    lines = [f"# Fine-tuning race{' (smoke)' if smoke else ''}", "",
             (f"{p['model']}, {os.path.basename(p['data'])}: {p['max_examples'] or p['n_train']} train / "
              f"{cfg['judge']['judge_n']} judge examples (split seed {p['split_seed']}), seq {p['seq']}, "
              f"micro-batch {p['batch']} x accum {p['accum']}, {p['epochs']} epoch, LR {p['lr']}, "
              f"LoRA r={p['lora_r']} alpha={p['lora_alpha']} dropout={p['lora_dropout']}, {p['threads']} threads, "
              f"seed {p['seed']}."), "",
             "Versions: " + ", ".join(f"{k} {v}" for k, v in summary["versions"].items() if v) + ".", "",
             f"Machine: {mach['cpu']}, {mach['usable_cpus']} CPUs, {fmt(mach['mem_total_gb'], '.1f')} GB, "
             f"load average {fmt(mach['loadavg'][0], '.1f')} at the end.", "",
             head + (" est. full train min |" if smoke else ""),
             "|---|---:|---:|---:|---:|---:|---:|---|---:|" + ("---:|" if smoke else "")]
    for n, a in arms.items():
        ci, rss = a["judge_ci95"], a.get("peak_rss_mb")
        lines.append(f"| {n} | {a.get('steps', '-')} | {fmt(_min(a.get('train_wall_s')), '.1f')} "
                     f"| {fmt(a.get('tokens_per_s'), '.0f')} | {fmt(_min(a.get('train_cpu_s')), '.1f')} "
                     f"| {fmt(rss and rss / 1024, '.2f')} | {fmt(a.get('final_train_loss'))} "
                     f"| {fmt(a['judge_loss'])} ({fmt(ci[0])}-{fmt(ci[1])}) | {fmt(a['judge_vs_base'], '+.3f')} |"
                     + (f" {fmt(a.get('est_full_train_min'), '.0f')} |" if smoke else ""))
    lines += ["", "Arms:", ""]
    for n, a in arms.items():
        s = a.get("settings") or {}
        extra = ""
        if s.get("optimizers"):
            extra = (f" Effective: optimizers {s['optimizers']}, state_bits {s.get('state_bits')}, "
                     f"pack {s.get('pack')}, chunked CE {s.get('chunked_ce')}, quant {s.get('quant')}"
                     + (f" (embeddings {s.get('embed_quant')})" if s.get("embed_quant") else "") + ".")
        if "judge_vs_hf_peft_ci95" in a:
            lo, hi = a["judge_vs_hf_peft_ci95"]
            extra += (f" Judge loss minus hf_peft: {a['judge_vs_hf_peft']:+.3f} "
                      f"(paired 95% CI {lo:+.3f} to {hi:+.3f}).")
        if "judge_loss_float_base" in a:
            extra += f" Its adapters on the float base: judge loss {fmt(a['judge_loss_float_base'])}."
        lines.append(f"- **{n}**: {a['desc']}.{extra}")
    sub = ", ".join(f"{n} {a['bantam_eval_minus_judge_subset']:+.1e}" for n, a in arms.items()
                    if "bantam_eval_minus_judge_subset" in a)
    trained = []
    for n, a in arms.items():
        if a.get("windows"):
            trained.append(f"{n} {a['supervised_tokens']} (in {a['windows_used']} of {a['windows']} packed windows)")
        elif a.get("supervised_tokens") is not None:
            trained.append(f"{n} {a['supervised_tokens']}")
    lines += ["", (f"Fairness checks: token parity {fair['token_parity']['mismatched']} mismatched of "
                   f"{fair['token_parity']['examples']}; same LoRA modules and trainable parameters in every arm: "
                   f"{fair['same_lora_modules']}; supervised tokens trained: " + ", ".join(trained)
                   + "; bantam finetune leaves out the windows that do not fill a last optimizer step; "
                   f"bantam's own final eval minus the judge on the same examples: {sub or '-'}."),
              "", "Commands:", "", "```"]
    lines += [f"# {n}\n{a['command']}" for n, a in arms.items() if a.get("command")]
    lines += ["```", "", "Greedy answers:", ""]
    for i, q in enumerate(cfg["judge"]["prompts"]):
        lines.append(f"*{q}*")
        lines += [f"- {n}: {a['generations'][i][:300]!r}" for n, a in arms.items()]
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in ("_hf", "_judge"):
        ap = argparse.ArgumentParser()
        ap.add_argument("--out", required=True)
        ap.add_argument("--arm", required=True)
        a = ap.parse_args(argv[1:])
        return (hf_worker if argv[0] == "_hf" else judge_worker)(a.out, a.arm)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=DEFAULT_ARMS, help=f"comma list of {', '.join(ARMS)}")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max_examples", type=int, default=0, help="train on the first N training examples (smoke)")
    ap.add_argument("--judge_n", type=int, default=200, help="judge the first N held-out examples")
    ap.add_argument("--out", default=os.environ.get("RACE_OUT") or os.path.join(ROOT, "results", "race"),
                    help="output dir (default: $RACE_OUT or results/race)")
    ap.add_argument("--force", default="", help="comma list of arms to rerun, or all")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--data_jsonl", default="", help="rows with instruction/context/response or prompt/response "
                    "instead of Dolly-15k from the HF cache")
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_judge", type=int, default=200)
    ap.add_argument("--split_seed", type=int, default=1234)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=float, default=0.03, help="warmup fraction of the optimizer steps")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=float, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--bantam_eval_n", type=int, default=32,
                    help="held-out examples bantam finetune evaluates itself (cross-checks the judge)")
    ap.add_argument("--gen_tokens", type=int, default=64)
    ap.add_argument("--judge_float_base", type=int, default=1, help="also judge quantized arms on the float base")
    ap.add_argument("--hf_flush_denormal", type=int, default=0,
                    help="turn on flush-to-zero in the HF arm too (bantam finetune always does)")
    ap.add_argument("--min_free_gb", type=float, default=5.0, help="warn when less memory is available")
    ap.add_argument("--online", action="store_true", help="allow Hugging Face Hub access (default: offline)")
    ap.add_argument("--dry_run", action="store_true", help="prepare the data and print the commands only")
    ap.add_argument("--quiet", action="store_true", help="do not echo the arms' output")
    args = ap.parse_args(argv)
    names = [a for a in args.arms.split(",") if a]
    bad = [a for a in names if a not in ARMS]
    if bad:
        ap.error(f"unknown arm(s) {bad}; choose from {', '.join(ARMS)}")
    args.force = {a for a in args.force.split(",") if a}
    args.out = os.path.abspath(args.out)
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    import fcntl
    lock = open(os.path.join(args.out, "race.lock"), "w")
    try:                                                   # released when this process exits
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"[race] another race is running in {args.out}") from None
    if not args.online:
        os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    state = {"versions": versions()}
    cfg = prepare(args, state)
    print(f"[race] {args.out}: arms {names}, {cfg['data']['full_train_examples']} full training examples "
          f"({cfg['data']['full_train_tokens']} tokens), token parity {cfg['token_parity']}", flush=True)
    if args.dry_run:
        for n in names:
            spec, run = ARMS[n], os.path.join(args.out, "runs", n)
            if spec["kind"] != "base":
                print(f"{n} train: " + shlex.join(bantam_cmd(cfg, spec, run) if spec["kind"] == "bantam"
                                                  else worker_cmd("_hf", args.out, n)))
            print(f"{n} judge: " + shlex.join(worker_cmd("_judge", args.out, n)))
        return 0
    failed = []
    for n in names:
        free = mem_available_gb()
        if free is not None and free < args.min_free_gb:
            print(f"[race] warning: {free:.1f} GB available (< --min_free_gb {args.min_free_gb})", flush=True)
        if run_arm(n, cfg, args, state).get("status") != "done":
            failed.append(n)
    summarize(args, cfg, state)
    print(f"[race] wrote {os.path.join(args.out, 'summary.md')}" + (f"; FAILED: {failed}" if failed else ""),
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
