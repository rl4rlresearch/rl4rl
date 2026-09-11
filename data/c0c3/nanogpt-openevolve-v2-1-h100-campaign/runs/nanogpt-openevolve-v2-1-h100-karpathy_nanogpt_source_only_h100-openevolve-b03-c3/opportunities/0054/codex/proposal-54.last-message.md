MECHANISM: Universal adaptive attention-output gating

HYPOTHESIS: Extending the proven 32-channel output gate to all eight attention layers, while retaining dedicated AdamW at 0.0115 and leaving value-residual gates on Muon, will lower val_bpb below 0.991177 while retaining at least 470M trained tokens.

INTENDED_EDIT: Replace handoff-only output gating with neutral-initialized output gating on every attention branch; the existing optimizer discovery automatically places all eight output-gate matrices in the dedicated AdamW group.

EVIDENCE: Dedicated AdamW for four output gates achieved the best observed 0.991177, whereas moving value-residual gates to AdamW regressed to 0.992169 and changing output-gate bias or input width also regressed; this motivates extending the successful gate form by attention-layer coverage without altering its parameterization or optimizer.

<<<<<<< SEARCH
        pattern = config.window_pattern.upper()
        pattern_idx = layer_idx % len(pattern)
        is_long_layer = pattern[pattern_idx] == "L" or layer_idx == config.n_layer - 1
        precedes_long_layer = pattern[(pattern_idx + 1) % len(pattern)] == "L"
        is_gated_layer = is_long_layer or precedes_long_layer
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_gated_layer else None
=======
        # Every attention branch receives the same neutral-initialized,
        # input-dependent output calibration that was successful at handoffs.
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
>>>>>>> REPLACE