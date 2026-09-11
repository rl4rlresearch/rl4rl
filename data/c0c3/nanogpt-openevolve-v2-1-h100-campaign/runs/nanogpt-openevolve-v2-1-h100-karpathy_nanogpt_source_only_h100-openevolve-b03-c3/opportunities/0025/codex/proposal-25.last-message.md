MECHANISM: Pre-long short-context attention gating

HYPOTHESIS: Restricting output gates to the two short-context layers immediately preceding full-context layers will lower val_bpb below 0.992110 while processing more than 477.1M tokens.

INTENDED_EDIT: Remove output gates from the full-context layers while retaining the proven two-lag embedding FIR and predecessor-layer gates.

EVIDENCE: Adding predecessor gates to the long-only design improved val_bpb from 0.992949 to 0.992110, whereas expanding to all layers reached only 0.992628; the missing predecessor-only ablation tests whether full-context gates are redundant with their existing value gates.

<<<<<<< SEARCH
        pattern = config.window_pattern.upper()
        pattern_idx = layer_idx % len(pattern)
        is_long_layer = pattern[pattern_idx] == "L" or layer_idx == config.n_layer - 1
        precedes_long_layer = pattern[(pattern_idx + 1) % len(pattern)] == "L"
        is_gated_layer = is_long_layer or precedes_long_layer
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_gated_layer else None
=======
        pattern = config.window_pattern.upper()
        pattern_idx = layer_idx % len(pattern)
        precedes_long_layer = pattern[(pattern_idx + 1) % len(pattern)] == "L"
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if precedes_long_layer else None
>>>>>>> REPLACE