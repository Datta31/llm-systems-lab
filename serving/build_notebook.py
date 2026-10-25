"""Builds serving_benchmark.ipynb. runner.py and analyze.py are embedded as %%writefile cells, so the single notebook
is all you upload to Colab or a GPU pod. Edit the .py files, then rebuild:
    ~/.venvs/serving/bin/python serving/build_notebook.py
"""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Serving benchmark: one 8B model, one GPU, five questions

Qwen3-8B served with **vLLM 0.32.0** and measured with `vllm bench serve`. Everything runs from a script in the
background and saves after every run, so a disconnect costs nothing: re-run the cells and it carries on.

| # | Question | Experiment | Lesson idea |
|---|---|---|---|
| 1 | How many users can one GPU serve, and how fast does each one feel it? | concurrency sweep 1 → 128 | batching trades per-user speed for throughput |
| 2 | Does a smaller number format pay off? | the same sweep in BF16, FP8, INT4 (AWQ) | fewer bytes per weight → faster decode, more KV cache |
| 3 | What do long agent prompts cost? | 8k-token prompts vs 1k | prefill dominates time to first token |
| 4 | What does a shared system prompt save? | prefix caching on vs off | same prefix → same K/V, computed once |
| 5 | When does speculative decoding help? | EAGLE-3 and n-gram, code vs chat, low vs high concurrency | spare compute at low batch; gone at high batch |

**GPU:** anything with compute capability ≥ 8.0 (L4, A100, L40S, H100). Not a T4. The plan sizes itself to the
GPU's memory. **Time, my estimate:** H100 ~1–1.5 h, A100 ~1.5–2 h, L4 ~3–4 h, of which ~10 min is setup.

**Before you start the GPU clock:** read sections 1–5 now; they're the same on any machine. Then run the cells top
to bottom. The only waits are the install (~3–5 min), the preflight (~5 min) and the main run.
""")

md("## 1. Machine check")
code("""
import json, os, subprocess, sys, time
from pathlib import Path

COLAB = "google.colab" in sys.modules
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(gpu or "No GPU visible. On Colab: Runtime → Change runtime type → L4 or A100.")
name, mem, cc, drv = [x.strip() for x in gpu.split(",")]
assert float(cc) >= 8.0, f"{name} (compute {cc}) lacks BF16 and modern kernels; pick an L4, A100, L40S or H100."
print("FP8 path:", "native W8A8 (Ada/Hopper tensor cores)" if float(cc) >= 8.9 else
      "weight-only (Marlin W8A16): Ampere has no FP8 tensor cores, so FP8 saves memory but not compute")
""")
code("""
# Where results go. On Colab, Google Drive, so they survive the runtime being recycled.
if COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    OUT = Path("/content/drive/MyDrive/llm-systems-lab/results-" + name.replace(" ", "_"))
    WORK = Path("/content/lab")
else:
    OUT = Path.home() / ("results-" + name.replace(" ", "_"))
    WORK = Path.home() / "lab"
OUT.mkdir(parents=True, exist_ok=True); WORK.mkdir(parents=True, exist_ok=True); os.chdir(WORK)
print("results →", OUT)
""")

md("""
## 2. Install vLLM and start the model downloads

vLLM runs as a separate process (the CLI), so it never touches this notebook's Python. The four repos (~32 GB) download
in the background while the install finishes. A Hugging Face token is optional (none of these models are gated); on
Colab, add one as the secret `HF_TOKEN` for faster, rate-limit-free downloads.
""")
code("""
# Installs into this kernel's own Python, whatever the image, and finds the CLIs next to it.
import shutil
PY = sys.executable
!{PY} -m pip install -q uv
!{PY} -m uv pip install --python {PY} -q "vllm[bench]==0.32.0" pandas matplotlib
BIN = Path(PY).parent
VLLM = shutil.which("vllm") or str(BIN / "vllm")
HF = shutil.which("hf") or str(BIN / "hf")
os.environ["BENCH_VLLM_BIN"] = VLLM
!{VLLM} --version
""")
code("""
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass
models = ["Qwen/Qwen3-8B", "Qwen/Qwen3-8B-FP8", "Qwen/Qwen3-8B-AWQ", "RedHatAI/Qwen3-8B-speculator.eagle3"]
dl = [subprocess.Popen([HF, "download", m], stdout=open(f"download-{m.split('/')[1]}.log", "w"),
                       stderr=subprocess.STDOUT) for m in models]
print("downloading in the background:", *models, sep="\\n  ")
""")
code("""
# Re-run to check. Each line says done or still going.
for m, p in zip(models, dl):
    print(f"{m:45s}", "done" if p.poll() == 0 else "still going" if p.poll() is None else f"FAILED, see download-{m.split('/')[1]}.log")
!df -h . | tail -1
""")

md("""
## 3. The code

`runner.py` starts one vLLM server per configuration, runs `vllm bench serve` against it at each concurrency level and
saves every run to `OUT/runs/`. Before and after each run it reads the server's `/metrics` counters, which is how it gets
the prefix-cache hit rate and the speculative-decoding acceptance length. `analyze.py` turns the folder into a CSV,
charts, a cost table and a findings draft. Both are tested on a laptop against a fake `vllm` (see `tests/`).
""")
code("%%writefile runner.py\n" + (HERE / "runner.py").read_text())
code("%%writefile analyze.py\n" + (HERE / "analyze.py").read_text())
code("""
!{PY} runner.py --out "{OUT}" --plan | tail -4
""")

md("""
## 4. Preflight (~5 min)

One server, one tiny run. It catches a broken install, a model that won't load or a benchmark flag that changed,
before the long run starts. The first start is the slowest: vLLM compiles the model and captures CUDA graphs.
""")
code("""
!{PY} runner.py --out "{OUT}" --preflight
print(open(OUT / "preflight/logs/server-bf16.log").read()[-1500:])
""")

md("""
## 5. The full run, in the background

It keeps going if this tab disconnects. To resume after a disconnect, run sections 1–3 again and then this cell:
finished runs are skipped. The next cell watches progress.
""")
code("""
progress = OUT / "progress.log"
start_line = len(progress.read_text().splitlines()) if progress.exists() else 0
run = subprocess.Popen(f"nohup '{PY}' runner.py --out '{OUT}' > runner.out 2>&1", shell=True, start_new_session=True)
time.sleep(5); print(open("runner.out").read())
""")
code("""
# Watch. Prints new progress lines every 30 s until the run finishes. Safe to stop and re-run.
seen = start_line
while True:
    lines = progress.read_text().splitlines() if progress.exists() else []
    for l in lines[seen:]: print(l)
    seen = len(lines)
    if any(l.split("  ", 1)[-1].startswith("finished") for l in lines[start_line:]): break
    if run.poll() is not None:
        print("runner stopped without finishing; its output:\\n" + open("runner.out").read()[-3000:]); break
    time.sleep(30)
""")

md("""
### While it runs: predict first, then check

Write down a guess for each, then compare with the charts. The why is in the inference notes
(`knowledge/llm-inference/` in career-copilot).

1. **Sweep.** From 1 to 128 concurrent requests, total tokens/s grows a lot but each user's tokens/s falls only a
   little at first. *Why:* decode is memory-bound. Each step reads all 16 GB of weights whatever the batch, so extra
   requests ride along almost free until compute runs out. Guess the concurrency where per-user speed drops below
   20 tokens/s.
2. **Precision.** FP8 should beat BF16 everywhere on an H100 or L4 (half the bytes to read, native FP8 maths). INT4 (AWQ)
   should win at concurrency 1 (a quarter of the bytes) but lose its lead at high concurrency, where it has to dequantize
   to 16-bit before every matmul and the GPU is compute-bound. Check the server table: fewer weight bytes also means
   more KV cache.
3. **Agent prompts.** An 8k prompt is 8× the prefill of a 1k one, so the time to first token rises roughly 8× at
   concurrency 1 and climbs faster with load, because new prompts wait behind other prefills. This is the cost
   structure of an agent loop.
4. **Prefix cache.** With 6,000 shared tokens out of 6,500, the cache skips ~90% of prefill, so the time to first token
   should drop several-fold. Throughput barely changes, because decode was never the problem here.
5. **Speculative decoding.** At concurrency 1, both methods should speed up code rewrites (predictable text) more than
   open questions. At high concurrency the speedup falls below 1.0: the GPU no longer has spare compute to check the
   guesses. τ (accepted tokens per step) tells you how good the guesses were.
""")

md("## 6. First look (on the GPU machine)")
code("""
!{PY} analyze.py "{OUT}"
from IPython.display import Image, Markdown, display
for f in sorted((OUT / "figures").glob("*.png")): display(Image(str(f)))
display(Markdown((OUT / "findings.md").read_text()))
""")

md("""
## 7. Save and stop paying

1. Results are in `OUT` (on Colab that's Google Drive already). The next cell also makes a zip to download.
2. **Stop the machine.** Colab: *Runtime → Disconnect and delete runtime* (an idle runtime still uses compute units).
   RunPod or JarvisLabs: stop **and delete/terminate** the instance; a stopped pod still bills for its disk.
""")
code("""
import shutil
z = shutil.make_archive(str(WORK / "serving-results"), "zip", OUT)
print(z)
if COLAB:
    from google.colab import files; files.download(z)
""")

md("""
## 8. After the GPU (on the Mac, no GPU needed)

Unzip into `llm-systems-lab/serving/results/<gpu>/`, then
`~/.venvs/serving/bin/python serving/analyze.py serving/results/<gpu>`. The write-up comes from `findings.md`,
rewritten in your own words, plus the four charts. The resume bullet takes its numbers from the cost table and
the speedups.
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                 "accelerator": "GPU", "colab": {"gpuType": "L4"}})
nbf.write(nb, HERE / "serving_benchmark.ipynb")
print("wrote", HERE / "serving_benchmark.ipynb", len(cells), "cells")
