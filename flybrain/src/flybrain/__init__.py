"""flybrain: MaleCNS connectome-constrained agent for FlyRunner.

Subpackages:
  pipeline  - neuPrint queries -> versioned subgraph artifact (Phase 2)
  model     - connectome-constrained rate/LIF substrate (Phase 3)
  env       - vectorized NumPy port of the FlyRunner engine as a Gymnasium env (Phase 4)
  train     - PPO / CMA-ES loops and baselines (Phase 4)
  bench     - backend benchmark (MLX vs PyTorch/MPS) informing the design decision
"""
__version__ = "0.1.0"
