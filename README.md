# llm-systems-lab

Hands-on LLM serving measurements, run on rented or Colab GPUs.

## serving/: one 8B model, one GPU, five questions

Qwen3-8B on vLLM 0.32.0, measured with `vllm bench serve`:

1. Throughput vs per-user speed as concurrency grows (1 → 128 requests)
2. BF16 vs FP8 vs INT4 (AWQ): speed, KV-cache headroom, cost per 1M tokens under a latency SLO
3. Agent-shaped 8k-token prompts vs 1k chat prompts
4. Prefix caching on vs off with a shared 6k-token system prompt
5. Speculative decoding (EAGLE-3 draft, n-gram lookup) on predictable vs open-ended text, at low and high concurrency

Run it: open `serving/serving_benchmark.ipynb` on a GPU machine (Colab L4/A100 or a pod with compute capability ≥ 8.0)
and run the cells top to bottom. The notebook embeds `runner.py` and `analyze.py`; rebuild it with
`python serving/build_notebook.py` after editing them. `serving/tests/fake_vllm.py` stands in for vLLM so the whole
flow can be tested without a GPU.

Results and write-up: *coming after the first run.*
