MECHANISM: Stronger projection-only Muon-rate compensation

HYPOTHESIS: Reducing only the 512×2560 MLP projection learning rate to 7/8 of the base rate will retain at least 445M training tokens and reduce val_bpb below 0.982763.

INTENDED_EDIT: Restore the throughput-optimal 5× MLP and apply a second equal log-space reduction to only its contraction matrices, leaving expansion and attention rates unchanged.

EVIDENCE: Projection-only sqrt(7/8) compensation achieved the best observed val_bpb, 0.982763, outperforming both the uncompensated 5× design at 0.982905 and expansion-only compensation at 0.983257; this tests whether the favorable projection-rate trend continues.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden_dim = 21 * config.n_embd // 4
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=False)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=False)
=======
class MLP(nn.Module):
    def __init__(self, config, layer_idx):
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
            # Continue the favorable projection-only LR trend by taking a
            # second equal log-space step beyond sqrt(7/8).
            mlp_lr_scale = 7 / 8 if shape[1] == 5 * shape[0] else 1.0
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr * mlp_lr_scale,
                momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=weight_decay,
            ))
>>>>>>> REPLACE