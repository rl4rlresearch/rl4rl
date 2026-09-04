MECHANISM: Ungated value-residual injection

HYPOTHESIS: Replacing the learned per-head value-embedding gate with its neutral fixed mixture will reduce val_bpb below 0.984227 by eliminating an unhelpful projection and increasing token throughput.

INTENDED_EDIT: Remove the 32-channel value-gate projection and inject each alternating value embedding directly into the attention value stream.

EVIDENCE: The prior attention-head gate experiment processed only 480.5M tokens and did not improve quality; fixing the gate at its initialization-equivalent value of one preserves the original value-residual signal while removing its measured overhead.

<<<<<<< SEARCH
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
=======
        self.ve_gate = None
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
=======
        # Value residual (ResFormer): fixed neutral mixture avoids gate overhead
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + ve
>>>>>>> REPLACE