#!/usr/bin/env python3
"""
Structural-syntax evaluation for the structural-token WaveGPT.

Tests three properties the user asked about, at the level the structural
tokenizer can actually express (identifiers are collapsed to ROLE tokens):

  1. Object retention  -> proxy: define-role vs use-role patterns. True
     co-reference is UNobservable in structural tokens (no per-identifier id).
  2. Scope-depth obligations -> bracket nesting () [] {}, INDENT/DEDENT
     balance, COLON->block(INDENT), binary-operator->operand, def->NAME_FUNC.
  3. Context collapse over position -> per-20-token windows: structural
     violation rate, repetition rate, token diversity, nesting depth.

Runs a grid of prompts x temperatures and prints per-gen + aggregate metrics.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from spectral.experiments.infer import load_from_checkpoint
from spectral.model.generate import _top_k_top_p_filter

CKPT = "checkpoints/wave_gpt_structural_1k/spectral_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Token category sets (from data/tokenizer.py vocabulary) ─────────────────
OPEN = {"LPAREN": "RPAREN", "LBRACKET": "RBRACKET", "LBRACE": "RBRACE"}
CLOSE = {v: k for k, v in OPEN.items()}
NAME_DEFINE = {"NAME_CLASS", "NAME_FUNC", "NAME_PARAM", "NAME_ASSIGN"}
NAME_USE = {"NAME", "NAME_CALL", "NAME_ATTR", "NAME_OTHER"}
NAME_ALL = NAME_DEFINE | NAME_USE
BIN_OPS = {"OP_ASSIGN", "OP_AUGASSIGN", "OP_COMPARE", "OP_WALRUS", "ARROW"}
OPERAND = NAME_ALL | {"NUMBER", "STRING", "FSTRING_START", "LPAREN", "LBRACKET",
                      "LBRACE", "True", "False", "None", "not", "lambda",
                      "await", "STAR", "DOUBLESTAR"}
TRIVIAL = {"COMMENT", "NL"}
TERMINALS = {"<EOS>", "ENDMARKER", "NEWLINE", "NL"}

PROMPTS = [
    "def ",
    "class A:",
    "for i in range(10):",
    "if x == 1:",
    "x = [",
    "d = {",
    "def f(x):\n    return ",
    "import ",
    "try:",
    "class A:\n    def __init__(self):\n        self.",
]
TEMPS = [0.0, 0.7, 1.0, 1.3]   # 0.0 == greedy/argmax
MAX_NEW = 80


@torch.no_grad()
def gen_names(model, tok, prompt, temperature, max_new=MAX_NEW, seed=0):
    torch.manual_seed(seed)
    if prompt.strip():
        ids = tok.encode(prompt, add_special=True)
        while len(ids) > 1 and tok.vocab[ids[-1]] in TERMINALS:
            ids = ids[:-1]
    else:
        ids = [tok.bos_id]
    n_prompt = len(ids)
    gen = list(ids)
    max_seq = getattr(model, "max_seq_len", 256)
    for _ in range(max_new):
        ctx = gen[-max_seq:]
        t = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
        p = torch.arange(len(ctx), dtype=torch.long, device=DEVICE).unsqueeze(0)
        logits = model(t, p, mode="lm")["lm_logits"][0, -1, :]
        if temperature <= 0:
            nxt = int(logits.argmax())
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        gen.append(nxt)
        if nxt == tok.eos_id:
            break
    return [tok.vocab[i] for i in gen], n_prompt


# ── Metrics ─────────────────────────────────────────────────────────────────
def delimiter_balance(names):
    stack, unmatched_close, matched, maxd = [], 0, 0, 0
    for t in names:
        if t in OPEN:
            stack.append(t); maxd = max(maxd, len(stack))
        elif t in CLOSE:
            if stack and stack[-1] == CLOSE[t]:
                stack.pop(); matched += 1
            else:
                unmatched_close += 1
    opens = sum(1 for t in names if t in OPEN)
    return dict(opens=opens, matched=matched, unmatched_close=unmatched_close,
                unclosed_open=len(stack), max_depth=maxd)


def indent_scope(names):
    depth = neg = maxd = 0
    for t in names:
        if t == "INDENT":
            depth += 1; maxd = max(maxd, depth)
        elif t == "DEDENT":
            if depth == 0:
                neg += 1
            else:
                depth -= 1
    return dict(neg_dedent=neg, residual=depth, max_depth=maxd)


def colon_block(names):
    sat = tot = 0
    for i, t in enumerate(names):
        if t != "COLON":
            continue
        if i + 1 < len(names) and names[i + 1] == "NEWLINE":   # block-opening colon
            tot += 1
            j = i + 2
            while j < len(names) and names[j] in TRIVIAL:
                j += 1
            if j < len(names) and names[j] == "INDENT":
                sat += 1
    return sat, tot


def binop_oblig(names):
    sat = tot = 0
    for i, t in enumerate(names):
        if t in BIN_OPS:
            tot += 1
            if i + 1 < len(names) and names[i + 1] in OPERAND:
                sat += 1
    return sat, tot


def kw_oblig(names):
    """def->NAME_FUNC, class->NAME_CLASS (allowing generic NAME)."""
    sat = tot = 0
    for i, t in enumerate(names):
        if t == "def":
            tot += 1; sat += (i + 1 < len(names) and names[i + 1] in ("NAME_FUNC", "NAME"))
        elif t == "class":
            tot += 1; sat += (i + 1 < len(names) and names[i + 1] in ("NAME_CLASS", "NAME"))
    return sat, tot


def violation_flags(names):
    """Per-index structural-violation booleans over the full sequence."""
    stack, depth = [], 0
    flags = [False] * len(names)
    for i, t in enumerate(names):
        bad = False
        if t in OPEN:
            stack.append(t)
        elif t in CLOSE:
            if stack and stack[-1] == CLOSE[t]:
                stack.pop()
            else:
                bad = True
        if t == "INDENT":
            depth += 1
        elif t == "DEDENT":
            if depth == 0:
                bad = True
            else:
                depth -= 1
        if t in BIN_OPS and not (i + 1 < len(names) and names[i + 1] in OPERAND):
            bad = True
        flags[i] = bad
    return flags


def windows(names, n_prompt, size=20):
    flags = violation_flags(names)
    rows = []
    gen = names[n_prompt:]
    gflags = flags[n_prompt:]
    for w in range((len(gen) + size - 1) // size):
        seg = gen[w * size:(w + 1) * size]
        segf = gflags[w * size:(w + 1) * size]
        if not seg:
            continue
        rep = sum(1 for k in range(1, len(seg)) if seg[k] == seg[k - 1])
        rows.append(dict(win=f"{w*size+1}-{w*size+len(seg)}", n=len(seg),
                         viol=sum(segf) / len(seg),
                         rep=rep / max(1, len(seg) - 1),
                         div=len(set(seg)) / len(seg)))
    return rows


def role_profile(names, n_prompt):
    gen = names[n_prompt:]
    d = sum(1 for t in gen if t in NAME_DEFINE)
    u = sum(1 for t in gen if t in NAME_USE)
    return d, u


def pct(a, b):
    return f"{100*a/b:4.0f}%" if b else "  - "


# ── Run ──────────────────────────────────────────────────────────────────────
def main():
    model, tok, cfg = load_from_checkpoint(str(REPO / CKPT), "auto")
    model.to(DEVICE).eval()
    print(f"\n{'='*78}\nMODEL: {CKPT}")
    print(f"  type={cfg.get('model_type')} tok={cfg.get('tokenizer_type','structural')} "
          f"d={cfg.get('d_model')} layers={cfg.get('n_layers')} "
          f"params={model.param_count()['total']:,} device={DEVICE}")
    print(f"  vocab={len(tok.vocab)} (structural roles)  max_new={MAX_NEW}")

    # aggregates per temperature
    agg = {tp: dict(delim_s=0, delim_t=0, ndd=0, resid=0, colon_s=0, colon_t=0,
                    bin_s=0, bin_t=0, kw_s=0, kw_t=0, defi=0, use=0,
                    wins={}) for tp in TEMPS}

    for tp in TEMPS:
        label = "greedy" if tp == 0.0 else f"T={tp}"
        print(f"\n{'#'*78}\n## TEMPERATURE: {label}\n{'#'*78}")
        print(f"{'prompt':<26} {'delim':>6} {'uncl':>4} {'-ded':>4} {'res':>3} "
              f"{'colon':>6} {'binop':>6} {'def/cls':>7} {'def:use':>8}")
        for pi, pr in enumerate(PROMPTS):
            names, n_prompt = gen_names(model, tok, pr, tp, seed=100 + pi)
            db = delimiter_balance(names)
            isc = indent_scope(names)
            cs, ct = colon_block(names)
            bs, bt = binop_oblig(names)
            ks, kt = kw_oblig(names)
            d, u = role_profile(names, n_prompt)
            a = agg[tp]
            a["delim_s"] += db["matched"]; a["delim_t"] += db["matched"] + db["unmatched_close"] + db["unclosed_open"]
            a["ndd"] += isc["neg_dedent"]; a["resid"] += isc["residual"]
            a["colon_s"] += cs; a["colon_t"] += ct
            a["bin_s"] += bs; a["bin_t"] += bt
            a["kw_s"] += ks; a["kw_t"] += kt
            a["defi"] += d; a["use"] += u
            for w in windows(names, n_prompt):
                wd = a["wins"].setdefault(w["win"], dict(n=0, viol=0.0, rep=0.0, div=0.0, c=0))
                wd["viol"] += w["viol"]; wd["rep"] += w["rep"]; wd["div"] += w["div"]; wd["c"] += 1
            prn = repr(pr)[:24]
            print(f"{prn:<26} {pct(db['matched'], db['matched']+db['unmatched_close']+db['unclosed_open']):>6} "
                  f"{db['unclosed_open']:>4} {isc['neg_dedent']:>4} {isc['residual']:>3} "
                  f"{pct(cs,ct):>6} {pct(bs,bt):>6} {pct(ks,kt):>7} {f'{d}:{u}':>8}")

        # show one representative token stream (the deep-nesting prompt) at this temp
        names, n_prompt = gen_names(model, tok, PROMPTS[-1], tp, seed=100 + len(PROMPTS) - 1)
        stream = " ".join(names[:n_prompt]) + "  ||  " + " ".join(names[n_prompt:n_prompt + 34])
        print(f"\n  sample (prompt || gen) @ {label}:\n    {stream}")

    # ── Aggregate summary ──
    print(f"\n\n{'='*78}\nAGGREGATE (all prompts) per temperature\n{'='*78}")
    print(f"{'metric':<34}" + "".join(f"{('greedy' if t==0 else 'T='+str(t)):>10}" for t in TEMPS))
    def line(label, fn):
        print(f"{label:<34}" + "".join(f"{fn(agg[t]):>10}" for t in TEMPS))
    line("delimiter matched %",     lambda a: pct(a['delim_s'], a['delim_t']))
    line("colon->INDENT %",         lambda a: pct(a['colon_s'], a['colon_t']))
    line("binop->operand %",        lambda a: pct(a['bin_s'], a['bin_t']))
    line("def/class->name %",       lambda a: pct(a['kw_s'], a['kw_t']))
    line("neg-dedent events (sum)", lambda a: str(a['ndd']))
    line("residual indent (sum)",   lambda a: str(a['resid']))
    line("define:use role ratio",   lambda a: f"{a['defi']}:{a['use']}")

    print(f"\n{'='*78}\nCONTEXT-COLLAPSE: per-20-token window, averaged over all prompts\n{'='*78}")
    for tp in TEMPS:
        label = "greedy" if tp == 0.0 else f"T={tp}"
        print(f"\n {label}:   {'window':<10}{'viol%':>8}{'repeat%':>9}{'diversity':>11}")
        for win in sorted(agg[tp]["wins"], key=lambda s: int(s.split('-')[0])):
            wd = agg[tp]["wins"][win]; c = wd["c"]
            print(f"            {win:<10}{100*wd['viol']/c:>7.1f}{100*wd['rep']/c:>9.1f}{wd['div']/c:>11.2f}")


if __name__ == "__main__":
    main()
