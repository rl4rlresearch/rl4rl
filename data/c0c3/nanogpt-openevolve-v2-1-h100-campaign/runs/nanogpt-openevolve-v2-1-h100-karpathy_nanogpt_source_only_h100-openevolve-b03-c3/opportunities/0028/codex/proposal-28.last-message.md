MECHANISM: First-handoff predecessor gating

HYPOTHESIS: Adding the first short-context predecessor gate to both full-context gates will lower val_bpb below 0.992110 while retaining more than 477M training tokens.

INTENDED_EDIT: Gate layers 2, 3, and 7: both full-context layers plus only the short-context layer preceding the first full-context transition.

EVIDENCE: Four handoff gates achieved 0.992110, versus 0.992949 for long-only gating; the final handoff pair alone regressed to 0.993029, indicating the first predecessor is the more informative missing gate to restore.

<<<<<<< SEARCH
        pattern = config.window_pattern.upper()
        is_long_layer = pattern[layer_idx % len(pattern)] == "L" or layer_idx == config.n_layer - 1
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_long_layer else None
=======
        pattern = config.window_pattern.upper()
        pattern_idx = layer_idx % len(pattern)
        is_long_layer = pattern[pattern_idx] == "L" or layer_idx == config.n_layer - 1
        first_long_idx = pattern.index("L")
        precedes_first_long = first_long_idx > 0 and layer_idx == first_long_idx - 1
        is_gated_layer = is_long_layer or precedes_first_long
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_gated_layer else None
>>>>>>> REPLACE