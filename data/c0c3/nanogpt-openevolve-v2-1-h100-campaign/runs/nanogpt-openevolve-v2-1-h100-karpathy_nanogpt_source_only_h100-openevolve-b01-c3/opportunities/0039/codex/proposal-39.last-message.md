MECHANISM: Aspect-compensated MLP Muon rate

HYPOTHESIS: Restoring the verified-best 5× MLP while compensating its Muon learning rate for aspect-ratio scaling will retain at least 445M training tokens and reduce val_bpb below 0.982905.

INTENDED_EDIT: Restore the 2560-channel MLP and scale only its Muon parameter groups by sqrt(4.375/5), preserving the effective matrix update magnitude previously tuned at 4.375× while leaving attention and gate rates unchanged.

EVIDENCE: The 5× MLP achieved the best val_bpb, 0.982905, while 4.875×, 5.125×, 5.25×, and 5.625× were worse. The optimizer additionally multiplies Muon rates by the square root of matrix aspect ratio, so widening the MLP silently raised its effective rate despite the 93.75% rate having been selected on the 4.375× design.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 41 * config.n_embd // 8
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
=======
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = 5 * config.n_embd
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
=======
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            # The fused Muon step scales LR by sqrt(matrix aspect ratio).
            # Compensate 5x MLP matrices back to the effective rate tuned at 4.375x.
            mlp_lr_scale = math.sqrt(7 / 8) if max(shape) == 5 * min(shape) else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
>>>>>>> REPLACE