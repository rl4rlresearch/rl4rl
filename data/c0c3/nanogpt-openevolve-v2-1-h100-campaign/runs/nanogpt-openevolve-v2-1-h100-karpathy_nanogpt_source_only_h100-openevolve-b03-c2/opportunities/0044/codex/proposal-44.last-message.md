MECHANISM: Stagewise context reallocation

HYPOTHESIS: Compute-neutral 256-token early local attention and 512-token late local attention, with full context at layers 4 and 8, will reduce val_bpb below 0.983618 by using early layers for cheap feature extraction and allocating wider context to later refinement.

INTENDED_EDIT: Restore the best shared 35% AdamW schedule and replace uniform short attention with a 256/512-token stagewise split whose average local window remains 384 tokens.

EVIDENCE: Uniform 384-token attention achieved the best val_bpb of 0.983618, while uniform 256 and 512 produced similar regressions of 0.984221 and 0.984156; combining those endpoints preserves the winning design’s attention compute while testing whether context is more valuable after the intermediate global layer. Removing that global layer regressed to 0.985116, so both verified full-context layers are retained.

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        early_short_window = long_window // 8
        late_short_window = long_window // 4
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            if char == "L":
                window_sizes.append((long_window, 0))
            else:
                short_window = early_short_window if layer_idx < config.n_layer // 2 else late_short_window
                window_sizes.append((short_window, 0))
        window_sizes[-1] = (long_window, 0)
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', schedule='scalar', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', schedule='scalar', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
=======
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows widen from 256 to 512
>>>>>>> REPLACE

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.35   # best verified cooldown for embeddings and lm_head
SCALAR_WARMDOWN_RATIO = 0.59 # align residual-mixing scalars with Muon matrices
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
=======
WARMDOWN_RATIO = 0.35   # best verified AdamW cooldown
MUON_WARMDOWN_RATIO = 0.59 # best verified Muon cooldown
>>>>>>> REPLACE

<<<<<<< SEARCH
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    scalar_lrm = get_lr_multiplier(progress, SCALAR_WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group_lrm = muon_lrm if group['kind'] == 'muon' else (
            scalar_lrm if group.get('schedule') == 'scalar' else lrm
        )
        group["lr"] = group["initial_lr"] * group_lrm
=======
    lrm = get_lr_multiplier(progress, WARMDOWN_RATIO)
    muon_lrm = get_lr_multiplier(progress, MUON_WARMDOWN_RATIO)
    muon_momentum = get_muon_momentum(step)
    muon_weight_decay = get_weight_decay(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * (muon_lrm if group['kind'] == 'muon' else lrm)
>>>>>>> REPLACE