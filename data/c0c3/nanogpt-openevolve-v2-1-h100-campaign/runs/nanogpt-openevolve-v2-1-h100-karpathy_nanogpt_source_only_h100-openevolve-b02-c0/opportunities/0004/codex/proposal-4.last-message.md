MECHANISM: Static-shape exhaustive kernel autotuning

HYPOTHESIS: Max-autotuning the compiled model and fused optimizer kernels will process more than the baseline’s 497.0M tokens while preserving its architecture, reducing val_bpb below 0.995558.

INTENDED_EDIT: Enable PyTorch’s max-autotune compilation mode for the model, AdamW update, and Muon update.

EVIDENCE: The baseline half-context architecture achieved the best result, while all three attention-pattern changes reduced either throughput or quality; optimizing its unchanged computation is the clearest next throughput test.

<<<<<<< SEARCH
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
=======
@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune")
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
>>>>>>> REPLACE

<<<<<<< SEARCH
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
=======
@torch.compile(dynamic=False, fullgraph=True, mode="max-autotune")
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
>>>>>>> REPLACE

<<<<<<< SEARCH
model = torch.compile(model, dynamic=False)
=======
model = torch.compile(model, dynamic=False, mode="max-autotune")
>>>>>>> REPLACE