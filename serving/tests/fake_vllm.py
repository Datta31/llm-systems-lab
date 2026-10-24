"""A stand-in for the `vllm` CLI so the runner and the analysis can be tested on a laptop without a GPU.

`serve` answers /health and /metrics and logs the same KV-cache line vLLM does; `bench serve` writes a result JSON with
the same keys as vLLM v0.32's, with made-up but plausibly shaped numbers. Set FAKE_FAIL=<substring of a run id> to make
that run fail. Never use its output as results.
"""
import json, math, os, random, re, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

STATE = Path(os.environ.get("FAKE_STATE", "/tmp/fake_vllm_state.json"))


def serve(args):
    model, extra = args[0], " ".join(args[1:])
    STATE.write_text(json.dumps({"model": model, "nocache": "no-enable-prefix-caching" in extra,
                                 "spec": "eagle3" if "eagle3" in extra else "ngram" if "ngram" in extra else None}))
    gib = 16.4 if model.endswith("8B") else 9.1 if "FP8" in model else 5.9
    kv = int((80 * 0.9 - gib - 4) * 1024**3 / 147456)
    print(f"Model loading took {gib} GiB memory and 12.3 seconds", flush=True)
    print(f"GPU KV cache size: {kv:,} tokens, Maximum concurrency for 16,384 tokens per request: {kv / 16384:.2f}x", flush=True)
    counts = {"prefix_cache_queries": 0.0, "prefix_cache_hits": 0.0, "spec_decode_num_drafts": 0.0,
              "spec_decode_num_accepted_tokens": 0.0, "num_preemptions": 0.0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path == "/metrics":
                for k in counts: counts[k] += random.uniform(100, 200)
                counts["prefix_cache_hits"] = counts["prefix_cache_queries"] * (0.05 if json.loads(STATE.read_text())["nocache"] else 0.8)
                body = "\n".join([f"# TYPE vllm:{k} counter"] + [f'vllm:{k}_total{{engine="0",model_name="m"}} {v}' for k, v in counts.items()])
            else:
                body = "ok"
            self.send_response(200); self.end_headers(); self.wfile.write(body.encode())

    time.sleep(1)
    HTTPServer(("127.0.0.1", int(args[args.index("--port") + 1])), H).serve_forever()


def bench(args):
    a = dict(zip(args[::2], args[1::2]))
    get = lambda k, d=None: args[args.index(k) + 1] if k in args else d
    fname = get("--result-filename")
    if os.environ.get("FAKE_FAIL") and os.environ["FAKE_FAIL"] in fname: sys.exit("simulated failure")
    st = json.loads(STATE.read_text())
    conc, n = int(get("--max-concurrency")), int(get("--num-prompts"))
    speed = {"Qwen/Qwen3-8B": 1.0, "Qwen/Qwen3-8B-FP8": 1.35, "Qwen/Qwen3-8B-AWQ": 1.25}[get("--model")]
    if "AWQ" in get("--model") and conc >= 32: speed = 0.85  # W4A16 dequant cost once compute-bound
    long_in = get("--random-input-len") == "8192" or get("--dataset-name") == "prefix_repetition"
    tpot = 7.0 / speed * (1 + conc / 40) * (1.4 if long_in else 1.0)
    if st["spec"]: tpot *= (0.55 if "code" in str(get("--dataset-path")) else 0.8) * (1 + conc / 50)
    ttft = (35 if not long_in else 260) / speed * (1 + conc / 12)
    if get("--dataset-name") == "prefix_repetition" and not st["nocache"]: ttft *= 0.25
    out_len = int(get("--random-output-len") or get("--prefix-repetition-output-len") or get("--custom-output-len") or 256)
    thr = conc * 1000 / tpot
    r = {"duration": n * out_len / thr, "completed": n, "failed": 0, "total_input_tokens": n * 1024,
         "total_output_tokens": n * out_len, "request_throughput": thr / out_len, "request_goodput": None,
         "output_throughput": thr, "total_token_throughput": thr * 5, "max_output_tokens_per_s": thr * 1.2,
         "max_concurrent_requests": conc}
    for m, base in [("ttft", ttft), ("tpot", tpot), ("itl", tpot), ("e2el", ttft + tpot * out_len)]:
        r[f"mean_{m}_ms"], r[f"median_{m}_ms"], r[f"std_{m}_ms"] = base * 1.05, base, base * 0.2
        for p, f in [("50", 1), ("90", 1.3), ("95", 1.45), ("99", 1.8)]: r[f"p{p}_{m}_ms"] = base * f
    r.update(backend="openai", model_id=get("--model"), num_prompts=n, max_concurrency=conc, date="fake")
    d = Path(get("--result-dir")); d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(json.dumps(r))
    print("fake bench done")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv[:1] == ["--version"]: print("0.32.0-fake")
    elif argv[:1] == ["serve"]: serve(argv[1:])
    elif argv[:2] == ["bench", "serve"]: bench(argv[2:])
    else: sys.exit(f"fake vllm: unknown {argv}")
