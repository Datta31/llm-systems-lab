# Findings draft (NVIDIA L4, vLLM 0.32.0)

Auto-generated from the numbers. Rewrite in your own words before publishing; check every claim against the charts.

## KV cache capacity (memory left after weights)

| server | weights GiB | KV cache tokens | startup s |
|---|---|---|---|
| bf16 | 15.27 | 22,960 | 270.1 |
| fp8 | 8.9 | 68,800 | 145.0 |
| int4 | 5.71 | 92,224 | 140.0 |
| bf16-nocache | 15.27 | 30,208 | 105.0 |
| bf16-eagle3 | 17.17 | 4,288 | 180.0 |
| bf16-ngram | 15.27 | 22,176 | 145.0 |

## Cost per 1M tokens at $0.44/GPU-hour (SLO: p95 TTFT ≤ 1 s chat / 3 s agent, p95 TPOT ≤ 50 ms)

| workload | precision | best_conc_in_slo | out_tok_s_in_slo | usd_per_1M_out_in_slo | usd_per_1M_total_in_slo | peak_out_tok_s | peak_conc | usd_per_1M_out_at_peak |
|---|---|---|---|---|---|---|---|---|
| chat | BF16 | – | – | – | – | 178 | 32 | 0.688 |
| chat | FP8 | 1.000 | 28.000 | 4.380 | 0.876 | 297 | 32 | 0.411 |
| chat | INT4 (AWQ) | 1.000 | 44.000 | 2.789 | 0.558 | 376 | 32 | 0.325 |
| agent | BF16 | – | – | – | – | 22 | 4 | 5.585 |
| agent | FP8 | – | – | – | – | 44 | 8 | 2.779 |
| agent | INT4 (AWQ) | 1.000 | 28.000 | 4.436 | 0.134 | 54 | 8 | 2.258 |

## Speculative decoding

| workload | method | concurrency | accept_len | tpot_speedup | throughput_ratio |
|---|---|---|---|---|---|
| spec-code | EAGLE-3 draft | 1 | 3.05 | 2.57 | 2.45 |
| spec-code | EAGLE-3 draft | 16 | 3.07 | 2.21 | 1.15 |
| spec-code | EAGLE-3 draft | 8 | 3.11 | 2.27 | 1.99 |
| spec-code | n-gram lookup | 1 | 2.51 | 2.43 | 2.26 |
| spec-code | n-gram lookup | 16 | 2.45 | 1.99 | 1.83 |
| spec-code | n-gram lookup | 8 | 2.57 | 2.25 | 2.01 |
| spec-chat | EAGLE-3 draft | 1 | 2.24 | 1.89 | 1.85 |
| spec-chat | EAGLE-3 draft | 16 | 2.24 | 1.74 | 1.59 |
| spec-chat | EAGLE-3 draft | 8 | 2.25 | 1.76 | 1.67 |
| spec-chat | n-gram lookup | 1 | 1.04 | 0.96 | 0.96 |
| spec-chat | n-gram lookup | 16 | 1.03 | 0.87 | 0.86 |
| spec-chat | n-gram lookup | 8 | 1.03 | 0.90 | 0.90 |

## Prefix caching

- concurrency 4: p50 TTFT 4040 ms → 912 ms with the cache (hit rate 71%)
- concurrency 8: p50 TTFT 21345 ms → 4382 ms with the cache (hit rate 72%)

## Preemptions (KV cache ran out; requests were paused and recomputed)

- bf16-eagle3__spec-chat__c16: 1
- bf16-eagle3__spec-code__c16: 31
- bf16-eagle3__spec-code__c8: 3
- bf16__chat__c32: 25
- bf16__prefix__c8: 6
