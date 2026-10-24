"""Serving benchmark runner: starts one vLLM server at a time and runs `vllm bench serve` against it.

Resume-safe: a run whose result file already exists is skipped, so after a disconnect you re-run the same command and
it carries on. Failed runs leave an `.error` file and are retried on the next start.

    python runner.py --out results            # the full plan for this GPU
    python runner.py --out results --preflight  # one tiny run (~3 min) to check the setup end to end
    python runner.py --out results --plan     # print the plan and exit
"""
import argparse, json, os, re, signal, subprocess, sys, time, urllib.request
from pathlib import Path

VLLM = os.environ.get("VLLM_BIN", "vllm")
PORT = int(os.environ.get("VLLM_PORT", 8000))
BASE = f"http://127.0.0.1:{PORT}"

BF16, FP8, INT4 = "Qwen/Qwen3-8B", "Qwen/Qwen3-8B-FP8", "Qwen/Qwen3-8B-AWQ"
DRAFT = "RedHatAI/Qwen3-8B-speculator.eagle3"
MODELS = [BF16, FP8, INT4, DRAFT]

# One server per configuration. All serve the same 8B model, so only the named setting differs.
SERVERS = {
    "bf16": (BF16, []),
    "fp8": (FP8, []),
    "int4": (INT4, []),
    "bf16-nocache": (BF16, ["--no-enable-prefix-caching"]),
    "bf16-eagle3": (BF16, ["--speculative-config", json.dumps(
        {"method": "eagle3", "model": DRAFT, "num_speculative_tokens": 3})]),
    "bf16-ngram": (BF16, ["--speculative-config", json.dumps(
        {"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 4})]),
}
# Context length and memory share per profile. On a 24 GB L4 the bf16 weights take ~15.3 GiB, which leaves room for
# roughly 25-30k tokens of KV cache, so 16k contexts may not fit; 12k still holds an 8k agent prompt.
SERVER_ARGS = {"large": ["--max-model-len", "16384", "--gpu-memory-utilization", "0.90"],
               "medium": ["--max-model-len", "16384", "--gpu-memory-utilization", "0.90"],
               "small": ["--max-model-len", "12288", "--gpu-memory-utilization", "0.92"]}

# Workloads. Random prompts with --ignore-eos give fixed lengths, so runs compare like for like.
WORKLOADS = {
    "chat": ["--dataset-name", "random", "--random-input-len", "1024", "--random-output-len", "256", "--ignore-eos"],
    # Shaped like an agent turn: a long prompt (system prompt, tool docs, results), a short answer.
    "agent": ["--dataset-name", "random", "--random-input-len", "8192", "--random-output-len", "256", "--ignore-eos"],
    # 4 shared 6,000-token prefixes (think: 4 agents' system prompts), each followed by a 500-token question.
    "prefix": ["--dataset-name", "prefix_repetition", "--prefix-repetition-prefix-len", "6000",
               "--prefix-repetition-suffix-len", "500", "--prefix-repetition-num-prefixes", "4",
               "--prefix-repetition-output-len", "128", "--ignore-eos"],
    # Real text, greedy, so the draft's guesses can actually be right. Thinking off to keep answers comparable.
    "spec-code": ["--dataset-name", "custom", "--dataset-path", "{prompts}/code_edit.jsonl",
                  "--custom-output-len", "256", "--temperature", "0",
                  "--chat-template-kwargs", '{"enable_thinking": false}'],
    "spec-chat": ["--dataset-name", "custom", "--dataset-path", "{prompts}/open_chat.jsonl",
                  "--custom-output-len", "256", "--temperature", "0",
                  "--chat-template-kwargs", '{"enable_thinking": false}'],
}

# Concurrency levels by GPU memory. A 24 GB L4 holds the 16 GB bf16 weights plus only ~4 GB of KV cache.
PROFILES = {
    "large": dict(chat=[1, 4, 16, 64, 128], agent=[1, 4, 16, 32], prefix=[8, 32], spec=[1, 16, 64]),  # 80 GB
    "medium": dict(chat=[1, 4, 16, 64], agent=[1, 4, 8, 16], prefix=[8, 16], spec=[1, 8, 32]),  # 40-48 GB
    "small": dict(chat=[1, 4, 16, 32], agent=[1, 2, 4, 8], prefix=[4, 8], spec=[1, 8, 16]),  # 24 GB
}
SEEDS = {w: 1000 * i for i, w in enumerate(WORKLOADS)}

# Server metrics read before and after each run (Prometheus counters gain a _total suffix).
COUNTERS = ["vllm:prefix_cache_queries", "vllm:prefix_cache_hits", "vllm:num_preemptions",
            "vllm:spec_decode_num_drafts", "vllm:spec_decode_num_draft_tokens", "vllm:spec_decode_num_accepted_tokens",
            "vllm:prompt_tokens", "vllm:generation_tokens"]


def gpu_info():
    if "GPU_INFO" in os.environ:  # tests on a machine without a GPU
        name, mem, cc = os.environ["GPU_INFO"].split(";")
    else:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.splitlines()[0]
        name, mem, cc = [x.strip() for x in out.split(",")]
    mem_gb = float(mem) / 1024
    return {"name": name, "memory_gb": round(mem_gb, 1), "compute_cap": cc,
            "profile": "large" if mem_gb >= 70 else "medium" if mem_gb >= 38 else "small"}


def plan(profile):
    """Every run, grouped by server so each server starts once."""
    p, runs = PROFILES[profile], []
    def add(server, workload, conc, n):
        runs.append({"id": f"{server}__{workload}__c{conc}", "server": server, "workload": workload,
                     "concurrency": conc, "num_prompts": n, "seed": SEEDS[workload] + conc})
    for s in ["bf16", "fp8", "int4"]:
        for c in p["chat"]: add(s, "chat", c, max(20, 4 * c))
        for c in p["agent"]: add(s, "agent", c, max(20, 4 * c))
    for s in ["bf16", "bf16-nocache"]:
        for c in p["prefix"]: add(s, "prefix", c, 128)
    for s in ["bf16", "bf16-eagle3", "bf16-ngram"]:
        for w in ["spec-code", "spec-chat"]:
            for c in p["spec"]: add(s, w, c, max(24, 3 * c))
    order = list(SERVERS)
    return sorted(runs, key=lambda r: order.index(r["server"]))


def log(out, msg):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(out / "progress.log", "a") as f: f.write(line + "\n")


def get(path, timeout=5):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r: return r.status, r.read().decode()


def scrape():
    """Sum each counter over its label sets."""
    _, text = get("/metrics")
    vals = dict.fromkeys(COUNTERS, 0.0)
    for line in text.splitlines():
        if line.startswith("#"): continue
        m = re.match(r"([a-zA-Z_:]+?)(_total)?(\{.*\})?\s+([0-9.eE+-]+)$", line)
        if m and m.group(1) in vals: vals[m.group(1)] += float(m.group(4))
    return vals


def start_server(name, out, profile):
    model, extra = SERVERS[name]
    import socket
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", PORT)) == 0:
            raise RuntimeError(f"port {PORT} is already in use (an old server?); stop it or set VLLM_PORT")
    logf = out / "logs" / f"server-{name}.log"
    args = list(SERVER_ARGS[profile])
    if name in ("bf16-eagle3", "bf16-ngram"):
        # Spec-decoding prompts are under 2k tokens. A short context leaves room for the 2 GB EAGLE-3 draft on a 24 GB L4.
        args[args.index("--max-model-len") + 1] = "4096"
    cmd = [VLLM, "serve", model, *args, "--port", str(PORT), *extra]
    proc = subprocess.Popen(cmd, stdout=open(logf, "w"), stderr=subprocess.STDOUT, start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < 1500:
        if proc.poll() is not None:
            raise RuntimeError(f"server {name} exited; last log lines:\n" + "".join(logf.read_text().splitlines(True)[-25:]))
        try:
            if get("/health", 2)[0] == 200: break
        except Exception: time.sleep(5)
    else:
        stop_server(proc); raise RuntimeError(f"server {name} not healthy after 25 min")
    text = logf.read_text()
    kv = re.search(r"KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", text)
    load = re.search(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", text)
    info = {"server": name, "model": model, "args": extra, "startup_s": round(time.time() - t0, 1),
            "kv_cache_tokens": int(kv.group(1).replace(",", "")) if kv else None,
            "kv_max_concurrency_at_max_len": float(kv.group(3)) if kv else None,
            "weights_gib": float(load.group(1)) if load else None}
    return proc, info


def stop_server(proc):
    try: os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError: return
    try: proc.wait(90)
    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
    time.sleep(5)  # let the GPU memory go before the next server starts


def bench(run, out, gpu):
    model = SERVERS[run["server"]][0]
    wl = [a.replace("{prompts}", str(out / "prompts")) for a in WORKLOADS[run["workload"]]]
    tmp = out / "raw"
    cmd = [VLLM, "bench", "serve", "--backend", "openai", "--base-url", BASE, "--model", model,
           "--num-prompts", str(run["num_prompts"]), "--max-concurrency", str(run["concurrency"]),
           "--seed", str(run["seed"]), "--num-warmups", "2",
           "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,90,95,99",
           "--save-result", "--result-dir", str(tmp), "--result-filename", run["id"] + ".json", *wl]
    before, t0 = scrape(), time.time()
    with open(out / "logs" / f"bench-{run['id']}.log", "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=1800)
    after = scrape()
    res = json.loads((tmp / (run["id"] + ".json")).read_text())
    res.update(run=run, gpu=gpu, wall_s=round(time.time() - t0, 1),
               server_counters={k: after[k] - before[k] for k in COUNTERS})
    (out / "runs" / (run["id"] + ".json")).write_text(json.dumps(res, indent=1))


def write_prompts(d):
    """Two prompt sets for speculative decoding: code rewrites (predictable output) and open questions."""
    import inspect, importlib
    d.mkdir(parents=True, exist_ok=True)
    funcs = []
    for mod in ["textwrap", "difflib", "statistics", "calendar", "fractions", "shlex", "csv", "argparse", "heapq",
                "ipaddress", "configparser", "string", "base64", "json.decoder", "json.encoder", "pprint"]:
        m = importlib.import_module(mod)
        for _, obj in sorted(vars(m).items()):
            if inspect.isfunction(obj) and obj.__module__ == m.__name__:
                try: src = inspect.getsource(obj)
                except OSError: continue
                if 15 <= src.count("\n") <= 60: funcs.append(src)
    funcs = funcs[:48]
    with open(d / "code_edit.jsonl", "w") as f:
        for src in funcs:
            f.write(json.dumps({"prompt": "Add Python type hints to this function and rename any one-letter "
                                "variables to descriptive names. Return only the full updated code.\n\n"
                                f"```python\n{src}```"}) + "\n")
    qs = ["Explain how a refrigerator keeps food cold", "Write a short story about a lighthouse keeper who finds a map",
          "What are the pros and cons of remote work for a small team", "Describe the water cycle to a ten-year-old",
          "Plan a three-day trip to Kyoto for someone who likes food and temples", "Why did the Roman Empire split in two",
          "Give advice to someone preparing for their first marathon", "Compare tea and coffee culture around the world",
          "Explain compound interest with an example", "Write a product description for a solar-powered backpack",
          "What makes a good team meeting", "Describe how vaccines train the immune system",
          "Write a letter from a cat to its owner about the new dog", "How do airplanes stay in the air",
          "Summarise the causes and effects of the 2008 financial crisis", "Suggest a weekly plan to learn guitar",
          "Explain what a black hole is and how we detect them", "Write a poem about monsoon rain in Mumbai",
          "What should a first-time manager focus on in their first 90 days", "How does a bill become law in India",
          "Describe the life cycle of a star", "Write a dialogue between a chef and a food critic",
          "Explain the difference between weather and climate", "How should a beginner start investing",
          "Describe a futuristic city in the year 2200", "What are the health benefits of regular sleep",
          "Explain how the internet routes a message across the world", "Write a speech for a school sports day",
          "Why do languages change over time", "Give tips for writing a clear email to a busy executive",
          "Describe how chess openings are learned and why they matter", "Explain photosynthesis step by step"]
    with open(d / "open_chat.jsonl", "w") as f:
        for q in qs: f.write(json.dumps({"prompt": q + ". Answer in about 300 words."}) + "\n")
    return len(funcs), len(qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--profile", choices=list(PROFILES))
    ap.add_argument("--servers", help="comma-separated subset, e.g. bf16,fp8")
    a = ap.parse_args()

    gpu = gpu_info()
    profile = a.profile or gpu["profile"]
    out = a.out / "preflight" if a.preflight else a.out
    for sub in ["runs", "raw", "logs"]: (out / sub).mkdir(parents=True, exist_ok=True)
    runs = ([{"id": "bf16__chat__c4", "server": "bf16", "workload": "chat", "concurrency": 4, "num_prompts": 8,
              "seed": 4}] if a.preflight else plan(profile))
    if a.servers: runs = [r for r in runs if r["server"] in a.servers.split(",")]
    if a.plan:
        for r in runs: print(r["id"], r["num_prompts"])
        print(f"{len(runs)} runs on {gpu['name']} ({profile})"); return

    write_prompts(out / "prompts")
    ver = subprocess.run([VLLM, "--version"], capture_output=True, text=True).stdout.strip()
    (out / "env.json").write_text(json.dumps({**gpu, "profile": profile, "vllm": ver}, indent=1))
    servers_file = out / "servers.json"
    servers = json.loads(servers_file.read_text()) if servers_file.exists() else {}

    todo = [r for r in runs if not (out / "runs" / (r["id"] + ".json")).exists()]
    log(out, f"{gpu['name']} · profile {profile} · vLLM {ver} · {len(runs) - len(todo)} done, {len(todo)} to run")
    failed = 0
    for name in dict.fromkeys(r["server"] for r in todo):
        group = [r for r in todo if r["server"] == name]
        log(out, f"starting server {name} ({len(group)} runs)")
        try:
            proc, info = start_server(name, out, profile)
        except RuntimeError as e:
            log(out, f"FAILED to start {name}: {e}"); failed += len(group); continue
        servers[name] = info; servers_file.write_text(json.dumps(servers, indent=1))
        log(out, f"  up in {info['startup_s']} s · KV cache {info['kv_cache_tokens']} tokens · weights {info['weights_gib']} GiB")
        try:
            for r in group:
                err = out / "runs" / (r["id"] + ".error")
                try:
                    bench(r, out, gpu); err.unlink(missing_ok=True)
                    res = json.loads((out / "runs" / (r["id"] + ".json")).read_text())
                    log(out, f"  {r['id']}: {res['output_throughput']:.0f} out tok/s · p50 TTFT {res['median_ttft_ms']:.0f} ms"
                             f" · p50 TPOT {res['median_tpot_ms']:.1f} ms · {res['completed']}/{r['num_prompts']} ok")
                except Exception as e:
                    failed += 1; err.write_text(repr(e)); log(out, f"  {r['id']}: FAILED ({e!r:.200}), see logs/")
        finally:
            stop_server(proc)
    log(out, f"finished · {failed} failed" + (" · re-run the same command to retry them" if failed else ""))


if __name__ == "__main__":
    main()
