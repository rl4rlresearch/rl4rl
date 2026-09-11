MECHANISM: Higher-sensitivity bounded value-residual gating

HYPOTHESIS: Doubling the local sensitivity of the existing bounded per-head value gate will lower val_bpb below 0.983689 by letting it adapt value-embedding contributions faster without increasing parameters or materially reducing throughput.

INTENDED_EDIT: Reparameterize the gate from `2*sigmoid(z)` to the equivalent-range but steeper `1+tanh(z)`, preserving neutral zero initialization and outputs bounded between 0 and 2.

EVIDENCE: Feature-wise value-residual gating regressed to 0.985862 and reduced training to 496.2M tokens, so retaining the efficient proven per-head pathway while improving its learning dynamics is more motivated than adding gate granularity.

<<<<<<< SEARCH
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
=======
        # Value residual (ResFormer): bounded per-head gate centered at neutral mixing
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 1 + torch.tanh(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
>>>>>>> REPLACE