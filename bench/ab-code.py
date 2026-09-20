#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fixed-prompt code decode bench, for A/B across server boots.

The numbers this box cares about are coding/agentic, and on this model decode
tracks draft acceptance almost linearly while acceptance is a property of the
prompt: bench/ notes measure a 20-30% spread between prompts, which is larger
than any knob being tested. So a comparison is only meaningful when both arms
run the SAME prompts, the same number of times, at the same temperature.

This script is that fixture. Prompts are frozen below; changing them
invalidates every arm recorded before the change.

    python3 bench/ab-code.py --arm baseline --out logs/ab.jsonl
    # relaunch with the knob changed
    python3 bench/ab-code.py --arm mamba-bf16 --out logs/ab.jsonl
    python3 bench/ab-code.py --compare logs/ab.jsonl

Requires an idle server: acceptance is read from vLLM's global /metrics
counters as a delta over the run, so concurrent traffic contaminates it.
Decode tok/s is per-request (usage.completion_tokens / first-delta->last-byte)
and is only mildly affected, but run it idle anyway.
"""
import argparse, json, os, statistics, time, urllib.request

BASE_DEFAULT = "http://localhost:8888"
MODEL = "qwen3.8-flash-next"

# Frozen. Five prompts, deliberately spread over the shapes the agent client
# actually sends: write-from-scratch, refactor, explain-and-patch, tests, and
# a structured/tool-shaped answer.
PROMPTS = [
    ("write", "Write a Python function `merge_intervals(intervals)` that merges "
              "overlapping closed intervals and returns them sorted. Include a "
              "docstring and handle the empty input. Code only."),
    ("refactor", "Refactor this into idiomatic Python with early returns and no "
                 "nested conditionals, and explain each change in one line:\n\n"
                 "def f(a, b):\n"
                 "    if a is not None:\n"
                 "        if b is not None:\n"
                 "            if a > b:\n"
                 "                return a - b\n"
                 "            else:\n"
                 "                return b - a\n"
                 "    return None\n"),
    ("explain", "Explain what a write-ahead log is, why a database needs one, "
                "and sketch the recovery procedure after a crash. Prose with a "
                "short numbered procedure at the end."),
    ("tests", "Write pytest tests for a function `parse_duration(s)` that accepts "
              "strings like '3h', '15m', '2h30m' and returns seconds, raising "
              "ValueError on anything else. Cover the error cases."),
    ("struct", "Return a JSON object describing three HTTP status codes (429, 502, "
               "504): for each, the name, when a client should retry, and a "
               "suggested backoff. JSON only, no prose."),
]

SPEC = ("vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_drafts_total")


def metrics(base):
    out = {}
    with urllib.request.urlopen(base + "/metrics", timeout=15) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("#"):
                continue
            for k in SPEC:
                if line.startswith(k + "{"):
                    out[k] = float(line.rsplit(" ", 1)[1])
    return out


def run_one(base, prompt, max_tokens, temperature):
    """Stream one completion; return (decode_tok_s, ttft_s, completion_tokens)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Explicit, so the arm does not depend on the server's default.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    usage = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            if not raw.startswith(b"data: "):
                continue
            body = raw[6:].strip()
            if body == b"[DONE]":
                break
            chunk = json.loads(body)
            if chunk.get("usage"):
                usage = chunk["usage"]
            ch = chunk.get("choices") or []
            if ch and (ch[0].get("delta") or {}).get("content") and first is None:
                first = time.perf_counter()
    last = time.perf_counter()
    if usage is None or first is None:
        raise RuntimeError("no usage or no content delta in stream")
    n = usage["completion_tokens"]
    # MTP means chunks != tokens; count from usage, never from chunk count.
    return n / (last - first), first - t0, n


def bench(args):
    before = metrics(args.base)
    rows = []
    for rep in range(args.reps):
        for name, prompt in PROMPTS:
            tok_s, ttft, n = run_one(args.base, prompt, args.max_tokens, args.temperature)
            rows.append({"prompt": name, "rep": rep, "tok_s": round(tok_s, 2),
                         "ttft_s": round(ttft, 3), "completion_tokens": n})
            print(f"  {name:9s} rep{rep}  {tok_s:6.2f} tok/s  ttft {ttft*1000:6.1f} ms  {n} tok")
    after = metrics(args.base)

    d_draft = after[SPEC[0]] - before[SPEC[0]]
    d_acc = after[SPEC[1]] - before[SPEC[1]]
    d_drafts = after[SPEC[2]] - before[SPEC[2]]
    rec = {
        "arm": args.arm,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reps": args.reps,
        "mean_tok_s": round(statistics.mean(r["tok_s"] for r in rows), 2),
        "median_tok_s": round(statistics.median(r["tok_s"] for r in rows), 2),
        "mean_ttft_ms": round(statistics.mean(r["ttft_s"] for r in rows) * 1000, 1),
        "acceptance_rate": round(d_acc / d_draft, 4) if d_draft else None,
        "accepted_per_draft": round(d_acc / d_drafts, 3) if d_drafts else None,
        "rows": rows,
    }
    print(f"\narm={rec['arm']}  mean {rec['mean_tok_s']} tok/s  "
          f"median {rec['median_tok_s']}  ttft {rec['mean_ttft_ms']} ms  "
          f"acceptance {rec['acceptance_rate']}  accepted/draft {rec['accepted_per_draft']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"appended to {args.out}")
    return rec


def compare(path):
    arms = {}
    for line in open(path):
        r = json.loads(line)
        arms.setdefault(r["arm"], []).append(r)
    print(f"{'arm':22s} {'runs':>4s} {'mean tok/s':>11s} {'median':>8s} "
          f"{'ttft ms':>8s} {'accept':>7s} {'acc/draft':>10s}")
    for arm, runs in arms.items():
        m = statistics.mean(r["mean_tok_s"] for r in runs)
        md = statistics.mean(r["median_tok_s"] for r in runs)
        tt = statistics.mean(r["mean_ttft_ms"] for r in runs)
        ac = [r["acceptance_rate"] for r in runs if r["acceptance_rate"]]
        ad = [r["accepted_per_draft"] for r in runs if r["accepted_per_draft"]]
        print(f"{arm:22s} {len(runs):4d} {m:11.2f} {md:8.2f} {tt:8.1f} "
              f"{statistics.mean(ac) if ac else float('nan'):7.3f} "
              f"{statistics.mean(ad) if ad else float('nan'):10.3f}")
    # Per-prompt, the comparison that actually controls for prompt choice.
    names = [n for n, _ in PROMPTS]
    print(f"\nper-prompt mean tok/s\n{'prompt':10s}" +
          "".join(f"{a:>16s}" for a in arms))
    for n in names:
        line = f"{n:10s}"
        for arm, runs in arms.items():
            vals = [row["tok_s"] for r in runs for row in r["rows"] if row["prompt"] == n]
            line += f"{statistics.mean(vals) if vals else float('nan'):16.2f}"
        print(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", default="unnamed", help="label for this server configuration")
    p.add_argument("--base", default=os.environ.get("BENCH_BASE", BASE_DEFAULT))
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--out", default="")
    p.add_argument("--compare", default="", help="read a jsonl and print the arm table")
    a = p.parse_args()
    if a.compare:
        compare(a.compare)
    else:
        bench(a)


if __name__ == "__main__":
    main()
