MECHANISM: Conservative adaptive optimization for handoff output gates

HYPOTHESIS: Optimizing only the four output-gate matrices with AdamW at 0.0075 on the proven 704/1024/1344 backbone will lower val_bpb below 0.991514 while retaining at least 470M trained tokens.

INTENDED_EDIT: Restore the best staged short-attention schedule, exclude output gates from Muon, and assign those gates a dedicated AdamW learning rate of 0.0075.

EVIDENCE: AdamW at 0.01 improved output-gated 704/1024/1344 from 0.991729 to the best observed 0.991514, while increasing it to 0.015 regressed to 0.991889; testing 0.0075 probes the more promising lower-learning-rate side while preserving the proven focus on output gates alone.

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
        assert pattern == "SSSL"
        # Preserve total short-attention FLOPs with the best verified
        # progression toward each full-context handoff.
        short_windows = (
            11 * long_window // 32,
            long_window // 2,
            21 * long_window // 32,
        )
        window_sizes = []
        for layer_idx in range(config.n_layer):
            pattern_idx = layer_idx % len(pattern)
            char = pattern[pattern_idx]
            window = long_window if char == "L" else short_windows[pattern_idx]
            window_sizes.append((window, 0))
        window_sizes[-1] = (long_window, 0)
        return window_sizes
>>>>>>> REPLACE

<<<<<<< SEARCH
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        context_params = [self.embedding_context]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) +
            len(x0_params) + len(context_params))
=======
        model_dim = self.config.n_embd
        output_gate_params = [
            block.attn.out_gate.weight for block in self.transformer.h
            if block.attn.out_gate is not None
        ]
        output_gate_ids = {id(p) for p in output_gate_params}
        matrix_params = [
            p for p in self.transformer.h.parameters()
            if id(p) not in output_gate_ids
        ]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        context_params = [self.embedding_context]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) +
            len(x0_params) + len(context_params) + len(output_gate_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=context_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
=======
            dict(kind='adamw', params=context_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=output_gate_params, lr=scalar_lr * 0.015, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
>>>>>>> REPLACE