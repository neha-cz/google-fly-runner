# Backend benchmark

batch=256, BPTT steps=32, ~10 in-edges/node, forward+backward per call, mean of 3 (torch 2.14.0).

| N nodes | torch CPU | torch MPS | MLX |
|---|---|---|---|
|   1000 | 105 ms (77,761 step·env/s) | 27 ms (300,638 step·env/s) | 17 ms (474,670 step·env/s) |
|   5000 | 1821 ms (4,498 step·env/s) | 1164 ms (7,035 step·env/s) | 626 ms (13,084 step·env/s) |
|  20000 | 12987 ms (631 step·env/s) | n/a | n/a |

## Reading the numbers

- Each "step" here is one Euler update of the whole batch; the real model uses ~13 sub-steps per
  15 Hz game decision, so divide by ~13 for env-decisions/s.
- Dense weight matrices are used up to 5k nodes (O(N²) per step); 20k uses torch sparse COO on CPU.
  MPS has no reliable sparse matmul and MLX has no sparse support, hence `n/a`.
- Decision recorded in `DESIGN.md` §7: PyTorch/MPS, graph sized at ~1–2k nodes, 5k hard cap.
- Regenerate with `cd flybrain && uv run flybrain bench-backend --out ../docs/bench-backend.md`.
