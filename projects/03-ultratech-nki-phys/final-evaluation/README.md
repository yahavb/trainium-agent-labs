# Frozen Final Evaluation

Two measured winners frozen by source SHA256. 32 unique new cases per task (64 total), seeds starting at 10000; development used 0..15. Exact duplicates of development inputs and earlier held-out inputs are excluded. Input NPZ bytes, checker/lowerer sources and manifest are pinned before execution. The original FP32 error gates are unchanged. No model is called or receives these results. This tests held-out values, not new equations, shapes or out-of-distribution conditions. No final results or speedups are claimed until device outputs are graded locally.
