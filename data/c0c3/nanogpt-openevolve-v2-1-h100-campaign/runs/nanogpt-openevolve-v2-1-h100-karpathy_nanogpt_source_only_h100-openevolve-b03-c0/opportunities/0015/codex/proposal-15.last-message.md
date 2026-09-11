MECHANISM: Ahead-of-time GEMM kernel autotuning

HYPOTHESIS: Free startup-time max-autotuning will raise throughput above 486.2M tokens while preserving the best model and optimization trajectory, achieving `val_bpb < 0.986636`.

INTENDED_EDIT: Compile the model with maximum kernel autotuning while disabling CUDA graphs to avoid additional memory pressure.

EVIDENCE: The best design reaches only 39.0% MFU, while depth and MLP reductions worsened validation quality; compilation is outside the measured window, so more aggressive kernel selection can improve throughput without sacrificing capacity.

<<<<<<< SEARCH
model = torch.compile(model, dynamic=False)
=======
model = torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")
>>>>>>> REPLACE