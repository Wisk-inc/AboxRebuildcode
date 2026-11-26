"""--recipe for `bantam pretrain` and `bantam weather`: the best settings measured so far,
applied only to the options the command line does not give.

They were measured at small scale on CPU (RESULTS.md): a 5.8M-parameter ZoomGPT on
TinyStories and a 1.26M-parameter ZoomViT on 5.625 deg ERA5, not larger models, other
data or GPUs. "none" (the default) changes nothing.
"""
import argparse
import sys

RECIPES = {
    "pretrain": {"best": {"optim": "muon", "lr": 1.6e-2, "qk_norm": 1,
                          "zoom": "2:0.33,1:0.67", "batch_scale": 2.0}},
    "weather": {"best": {"optim": "muon", "lr": 1.6e-2, "coarse_mode": "lowres",
                         "zoom": "4:0.6,1:0.4"}},
}
NAMES = ("none", "best")
# measured together: the recipe's --lr is applied only with the recipe's --optim
TIED = {"lr": "optim"}


def given_options(parser, argv):
    """dests of the options argv gives explicitly (also at their default value, abbreviated or
    as --opt=value): argv parsed again by `parser` with every default suppressed. `parser`
    is a fresh copy: its actions are changed."""
    for a in parser._actions:
        a.default, a.required = argparse.SUPPRESS, False
    return set(vars(parser.parse_known_args(argv)[0]))


def apply_recipe(args, make_parser, argv, command):
    """Set the --recipe options that argv does not give (args.recipe_applied lists them, in
    recipe order) and return a one-line note for the log, or None for --recipe none.
    make_parser() builds the command's parser (parsed again to see which options argv gives)."""
    args.recipe_applied = []
    if args.recipe == "none":
        return None
    rec = RECIPES[command][args.recipe]
    given = given_options(make_parser(), sys.argv[1:] if argv is None else list(argv))
    skipped = []
    for k, v in rec.items():
        if k in given:
            continue
        dep = TIED.get(k)
        if dep in given and getattr(args, dep) != rec[dep]:
            skipped.append(f"--{k} (measured with --{dep} {rec[dep]})")
            continue
        setattr(args, k, v)
        args.recipe_applied.append(k)
    note = f"--recipe {args.recipe}: " + (" ".join(f"--{k} {getattr(args, k)}"
                                                   for k in args.recipe_applied) or "nothing set")
    kept = [f"--{k}" for k in rec if k in given]
    if kept:
        note += "; from the command line: " + " ".join(kept)
    if skipped:
        note += "; not set: " + ", ".join(skipped)
    return note
