MECHANISM: Feature-wise value-residual gating

HYPOTHESIS: Replacing each value embedding’s per-head scalar gate with a per-feature gate will reduce val_bpb below 0.983689 by enabling finer token-dependent routing with negligible parameter and compute growth.

INTENDED_EDIT: Expand the existing zero-initialized value-residual gate from one coefficient per KV head to one coefficient per KV feature, retaining the SMMLSSSL attention schedule and neutral initialization.

EVIDENCE: Attention-window reallocations have plateaued at 0.983689, while parameter-matched SwiGLU regressed to 0.986926 alongside substantially lower throughput; this motivates adding expressivity specifically to the existing value-residual pathway at much lower cost.

<<<<<<< SEARCH
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
=======
        self.ve_gate = nn.Linear(
            self.ve_gate_channels, self.n_kv_head * self.head_dim, bias=False
        ) if has_ve(layer_idx, config.n_layer) else None
>>>>>>> REPLACE

<<<<<<< SEARCH
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
=======
            gate = 2 * torch.sigmoid(
                self.ve_gate(x[..., :self.ve_gate_channels]).view(
                    B, T, self.n_kv_head, self.head_dim
                )
            )
            v = v + gate * ve
>>>>>>> REPLACE