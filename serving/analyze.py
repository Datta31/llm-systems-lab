"""Turns a results folder from runner.py into a table, four charts, a cost table and a findings draft.

    python analyze.py results            # writes results/results.csv, results/figures/*.png, results/findings.md
    python analyze.py results --price 2.69   # $/GPU-hour if the GPU isn't in PRICES
"""
import argparse, json, math
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# $/GPU-hour, RunPod Community Cloud list prices on 2026-10-25 (runpod.io/pricing). Colab bills in compute units, so
# cost is always computed at what renting the same GPU would cost.
PRICES = [("H100 NVL", 2.59), ("H100 PCIe", 1.99), ("H100", 2.69), ("H200", 3.59), ("A100-SXM4-80GB", 1.39),
          ("A100 80GB PCIe", 1.19), ("L40S", 0.79), ("L4", 0.44)]
SLO = {"chat": {"ttft": 1000, "tpot": 50}, "agent": {"ttft": 3000, "tpot": 50}}  # p95, ms. 50 ms/token = 20 tok/s

BLUE, ORANGE, GREEN, GRAY, INK2, GRID = "#2a78d6", "#eb6834", "#2e9e6a", "#8a8984", "#52514e", "#e4e3df"
COLOR = {"bf16": BLUE, "fp8": ORANGE, "int4": GREEN, "bf16-eagle3": ORANGE, "bf16-ngram": GREEN, "bf16-nocache": GRAY}
LABEL = {"bf16": "BF16", "fp8": "FP8", "int4": "INT4 (AWQ)", "bf16-eagle3": "EAGLE-3 draft", "bf16-ngram": "n-gram lookup",
         "bf16-nocache": "prefix cache off"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.color": GRID, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 7})


def load(d):
    rows = []
    for f in sorted((d / "runs").glob("*.json")):
        r = json.loads(f.read_text()); run, c = r["run"], r.get("server_counters", {})
        drafts = c.get("vllm:spec_decode_num_drafts", 0)
        q = c.get("vllm:prefix_cache_queries", 0)
        rows.append({**{k: run[k] for k in ["id", "server", "workload", "concurrency", "num_prompts"]},
                     "completed": r["completed"], "failed": r.get("failed", 0),
                     "out_tok_s": r["output_throughput"], "total_tok_s": r["total_token_throughput"],
                     "req_s": r["request_throughput"],
                     **{f"{p}_{m}": r.get(f"{'median' if p == 'p50' else p}_{m}_ms") for m in ["ttft", "tpot", "itl", "e2el"]
                        for p in ["p50", "p95", "p99"]},
                     "user_tok_s": 1000 / r["median_tpot_ms"] if r.get("median_tpot_ms") else None,
                     "prefix_hit_rate": c.get("vllm:prefix_cache_hits", 0) / q if q else None,
                     "accept_len": 1 + c.get("vllm:spec_decode_num_accepted_tokens", 0) / drafts if drafts else None,
                     "preemptions": c.get("vllm:num_preemptions", 0)})
    env = json.loads((d / "env.json").read_text())
    servers = json.loads((d / "servers.json").read_text()) if (d / "servers.json").exists() else {}
    return pd.DataFrame(rows), env, servers


def price_for(gpu, override):
    if override: return override
    return next((p for k, p in PRICES if k in gpu), None)


def fig_frontier(df, out, gpu):
    """Throughput per GPU vs speed each user sees: the serving trade-off in one picture."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, wl in zip(axes, ["chat", "agent"]):
        for s in ["bf16", "fp8", "int4"]:
            d = df[(df.server == s) & (df.workload == wl)].sort_values("concurrency")
            if d.empty: continue
            ax.plot(d.user_tok_s, d.out_tok_s, "o-", color=COLOR[s], label=LABEL[s])
            for _, r in d.iterrows():
                ax.annotate(f"{r.concurrency}", (r.user_tok_s, r.out_tok_s), textcoords="offset points",
                            xytext=(5, 4), fontsize=8, color=INK2)
        ax.axvline(1000 / SLO[wl]["tpot"], color=GRAY, ls="--", lw=1)
        ax.set(title=f"{wl}: {'1k in / 256 out' if wl == 'chat' else '8k in / 256 out'}",
               xlabel="tokens/s each user sees (1000 / p50 TPOT)", ylabel="output tokens/s per GPU")
    axes[0].legend(frameon=False)
    fig.suptitle(f"Throughput vs per-user speed, Qwen3-8B on {gpu} (labels = concurrent requests; dashed = 20 tok/s)",
                 fontsize=10, color=INK2)
    fig.tight_layout(); fig.savefig(out / "1_frontier.png"); plt.close(fig)


def fig_ttft(df, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in ["bf16", "fp8", "int4"]:
        d = df[(df.server == s) & (df.workload == "agent")].sort_values("concurrency")
        if not d.empty: ax.plot(d.concurrency, d.p50_ttft, "o-", color=COLOR[s], label=LABEL[s])
    ax.set_xscale("log", base=2)
    ax.set(title="Agent-shaped prompts (8k tokens): time to first token", xlabel="concurrent requests",
           ylabel="p50 TTFT (ms)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "2_ttft_agent.png"); plt.close(fig)


def fig_prefix(df, out):
    d = df[df.workload == "prefix"]
    if d.empty: return
    concs = sorted(d.concurrency.unique()); w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, s in enumerate(["bf16-nocache", "bf16"]):
        dd = d[d.server == s].set_index("concurrency").reindex(concs)
        xs = [j + (i - 0.5) * w for j in range(len(concs))]
        ax.bar(xs, dd.p50_ttft, w * 0.92, color=COLOR[s] if s != "bf16" else BLUE,
               label="prefix cache off" if s == "bf16-nocache" else "prefix cache on (default)")
        for x, v, h in zip(xs, dd.p50_ttft, dd.prefix_hit_rate):
            if not math.isnan(v):
                ax.annotate(f"{v:.0f} ms\nhit {h:.0%}" if h == h and h is not None else f"{v:.0f} ms", (x, v),
                            ha="center", va="bottom", fontsize=8, color=INK2)
    ax.margins(y=0.18)
    ax.set_xticks(range(len(concs)), [str(c) for c in concs])
    ax.set(title="6k-token shared prefix + 500-token question", xlabel="concurrent requests", ylabel="p50 TTFT (ms)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(out / "3_prefix_cache.png"); plt.close(fig)


def spec_table(df):
    rows = []
    for wl in ["spec-code", "spec-chat"]:
        base = df[(df.server == "bf16") & (df.workload == wl)].set_index("concurrency")
        for s in ["bf16-eagle3", "bf16-ngram"]:
            for _, r in df[(df.server == s) & (df.workload == wl)].iterrows():
                if r.concurrency in base.index:
                    b = base.loc[r.concurrency]
                    rows.append({"workload": wl, "method": LABEL[s], "server": s, "concurrency": r.concurrency,
                                 "accept_len": r.accept_len, "tpot_speedup": b.p50_tpot / r.p50_tpot,
                                 "throughput_ratio": r.out_tok_s / b.out_tok_s})
    return pd.DataFrame(rows)


def fig_spec(st, out):
    if st.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, wl in zip(axes, ["spec-code", "spec-chat"]):
        for s in ["bf16-eagle3", "bf16-ngram"]:
            d = st[(st.server == s) & (st.workload == wl)].sort_values("concurrency")
            if d.empty: continue
            ax.plot(d.concurrency, d.tpot_speedup, "o-", color=COLOR[s], label=LABEL[s])
            for _, r in d.iterrows():
                if r.accept_len == r.accept_len and r.accept_len:
                    ax.annotate(f"τ={r.accept_len:.2f}", (r.concurrency, r.tpot_speedup), textcoords="offset points",
                                xytext=(5, 5), fontsize=8, color=INK2)
        ax.axhline(1, color=GRAY, ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set(title="code rewrites (predictable output)" if wl == "spec-code" else "open questions",
               xlabel="concurrent requests", ylabel="per-user speedup vs no speculation (p50 TPOT)")
    axes[0].legend(frameon=False)
    fig.suptitle("Speculative decoding: τ = mean tokens accepted per step; below 1.0 = slower", fontsize=10, color=INK2)
    fig.tight_layout(); fig.savefig(out / "4_speculative.png"); plt.close(fig)


def cost_table(df, price):
    rows = []
    for wl in ["chat", "agent"]:
        for s in ["bf16", "fp8", "int4"]:
            d = df[(df.server == s) & (df.workload == wl)]
            if d.empty: continue
            ok = d[(d.p95_ttft <= SLO[wl]["ttft"]) & (d.p95_tpot <= SLO[wl]["tpot"])]
            best = ok.loc[ok.out_tok_s.idxmax()] if not ok.empty else None
            peak = d.loc[d.out_tok_s.idxmax()]
            usd = lambda tps: price / (tps * 3600) * 1e6 if price and tps else None
            rows.append({"workload": wl, "precision": LABEL[s],
                         "best_conc_in_slo": int(best.concurrency) if best is not None else None,
                         "out_tok_s_in_slo": round(best.out_tok_s) if best is not None else None,
                         "usd_per_1M_out_in_slo": usd(best.out_tok_s) if best is not None else None,
                         "usd_per_1M_total_in_slo": usd(best.total_tok_s) if best is not None else None,
                         "peak_out_tok_s": round(peak.out_tok_s), "peak_conc": int(peak.concurrency),
                         "usd_per_1M_out_at_peak": usd(peak.out_tok_s)})
    return pd.DataFrame(rows)


def md(df, digits=2):
    """A markdown table without the tabulate dependency."""
    fmt = lambda v: "–" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}" if isinstance(v, float) else str(v)
    return "\n".join(["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)])


def findings(df, env, servers, ct, st, price):
    L = [f"# Findings draft ({env['name']}, vLLM {env['vllm']})", "",
         "Auto-generated from the numbers. Rewrite in your own words before publishing; check every claim against the charts.", ""]
    L += ["## KV cache capacity (memory left after weights)", "", "| server | weights GiB | KV cache tokens | startup s |",
          "|---|---|---|---|"]
    for n, s in servers.items():
        L.append(f"| {n} | {s.get('weights_gib')} | {s.get('kv_cache_tokens'):,} | {s.get('startup_s')} |"
                 if s.get("kv_cache_tokens") else f"| {n} | {s.get('weights_gib')} | ? | {s.get('startup_s')} |")
    L += ["", f"## Cost per 1M tokens at ${price}/GPU-hour (SLO: p95 TTFT ≤ 1 s chat / 3 s agent, p95 TPOT ≤ 50 ms)", "",
          md(ct, 3) if not ct.empty else "(no sweep data)", ""]
    if not st.empty:
        L += ["## Speculative decoding", "", md(st.drop(columns="server")), ""]
    p = df[df.workload == "prefix"]
    if not p.empty:
        L += ["## Prefix caching", ""]
        for c in sorted(p.concurrency.unique()):
            on = p[(p.server == "bf16") & (p.concurrency == c)]; off = p[(p.server == "bf16-nocache") & (p.concurrency == c)]
            if len(on) and len(off):
                L.append(f"- concurrency {c}: p50 TTFT {off.p50_ttft.iloc[0]:.0f} ms → {on.p50_ttft.iloc[0]:.0f} ms with the cache "
                         f"(hit rate {on.prefix_hit_rate.iloc[0]:.0%})")
        L.append("")
    pre = df[df.preemptions > 0]
    if not pre.empty:
        L += ["## Preemptions (KV cache ran out; requests were paused and recomputed)", "",
              *[f"- {r.id}: {r.preemptions:.0f}" for r in pre.itertuples()], ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--price", type=float, help="$/GPU-hour override")
    a = ap.parse_args()
    df, env, servers = load(a.results)
    out = a.results / "figures"; out.mkdir(exist_ok=True)
    price = price_for(env["name"], a.price)
    df.to_csv(a.results / "results.csv", index=False)
    fig_frontier(df, out, env["name"]); fig_ttft(df, out); fig_prefix(df, out)
    st = spec_table(df); fig_spec(st, out)
    ct = cost_table(df, price)
    (a.results / "findings.md").write_text(findings(df, env, servers, ct, st, price))
    print(f"{len(df)} runs · {env['name']} · ${price}/h")
    print(ct.to_string(index=False))
    print(f"wrote {a.results}/results.csv, findings.md, figures/")


if __name__ == "__main__":
    main()
