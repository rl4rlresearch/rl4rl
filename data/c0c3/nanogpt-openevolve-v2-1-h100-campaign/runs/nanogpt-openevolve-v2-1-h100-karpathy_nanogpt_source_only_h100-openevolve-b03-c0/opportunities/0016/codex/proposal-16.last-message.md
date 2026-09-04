MECHANISM: CUDA-graph replay with maximum kernel autotuning

HYPOTHESIS: Enabling CUDA graphs alongside max-autotuning will exceed 492.1M training tokens while preserving the model and optimizer trajectory, achieving `val_bpb < 0.986491`.

INTENDED_EDIT: Compile the model in `max-autotune` mode, enabling CUDA-graph capture instead of explicitly disabling it.

EVIDENCE: Max-autotuning without CUDA graphs improved throughput from 486.2M to 492.1M tokens and `val_bpb` from 0.986636 to 0.986491; its 36.9GB peak allocation leaves substantial H100 memory headroom for graph capture.

<<<<<<< SEARCH
model = torch.compile(model, dynamic=False, mode="max-autotune-no-cudagraphs")
=======
model = torch.compile(model, dynamic=False, mode="max-autotune")
>>>>>>> REPLACE