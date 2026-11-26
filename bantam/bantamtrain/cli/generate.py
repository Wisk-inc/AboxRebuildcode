"""Generate text with a ZoomGPT saved by `ZoomGPT.save_pretrained`.

    bantam generate --model DIR --prompt "Once upon a time" \
        [--tokenizer DATA/tokenizer.json] [--max_new 200 --temperature 0.8 --top_k 40 --top_p 0.95]

The tokenizer defaults to DIR/tokenizer.json (save_pretrained(tokenizer_path=...)
copies it from the data dir). By default the prompt starts a new document, i.e.
it follows the end-of-document token as every training document does, and
decoding stops at that token.
"""
import argparse
import json
import os
import sys

import torch


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bantam generate", description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="directory written by save_pretrained")
    ap.add_argument("--tokenizer", default="",
                    help="tokenizer.json (default: MODEL/tokenizer.json)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8, help="<= 0: greedy")
    ap.add_argument("--top_k", type=int, default=40, help="0: off")
    ap.add_argument("--top_p", type=float, default=0.95, help="1: off")
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--max_ctx", type=int, default=0,
                    help="attention window (0: the model's block_size)")
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eos_id", type=int, default=None,
                    help="end-of-document id (default: config.json, else the tokenizer's <|eos|>)")
    ap.add_argument("--doc_start", type=int, default=1,
                    help="1: put the end-of-document token before the prompt")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if args.num_samples < 1:
        ap.error("--num_samples must be >= 1")
    if not os.path.isfile(os.path.join(args.model, "config.json")):
        ap.error(f"--model {args.model}: not a directory written by save_pretrained (no "
                 "config.json); `bantam pretrain --save X.pt` writes it as X_pretrained/")
    try:
        from tokenizers import Tokenizer
    except ImportError:
        raise SystemExit("bantam generate needs the tokenizers package: "
                         "pip install 'bantamtrain[data]'") from None
    from ..models import ZoomGPT

    tok_path = args.tokenizer or os.path.join(args.model, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit(f"no tokenizer at {tok_path}: pass --tokenizer DATA_DIR/tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    model = ZoomGPT.from_pretrained(args.model, map_location=args.device)
    eos = args.eos_id
    if eos is None:
        with open(os.path.join(args.model, "config.json")) as f:
            eos = json.load(f).get("eos_id", tok.token_to_id("<|eos|>"))
    ids = tok.encode(args.prompt).ids
    if args.doc_start and eos is not None:
        ids = [eos] + ids
    if not ids:
        raise SystemExit("empty prompt (and no end-of-document token to start from)")
    gen = torch.Generator(device=model.emb.weight.device).manual_seed(args.seed)
    out = model.generate([ids] * args.num_samples, args.max_new, temperature=args.temperature,
                         top_k=args.top_k or None, top_p=args.top_p, eos_id=eos,
                         repetition_penalty=args.repetition_penalty,
                         max_ctx=args.max_ctx or None, generator=gen)
    for i, row in enumerate(out):
        new = row[len(ids):]
        if eos is not None and new and new[-1] == eos:
            new = new[:-1]
        if i:
            print("-" * 40)
        print(args.prompt + tok.decode(new), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
