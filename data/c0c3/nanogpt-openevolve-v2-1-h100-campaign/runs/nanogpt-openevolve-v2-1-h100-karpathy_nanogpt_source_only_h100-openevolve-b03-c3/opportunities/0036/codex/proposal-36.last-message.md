MECHANISM: Fine-grained compute-neutral staged attention

HYPOTHESIS: A 704/1024/1344-token progression will lower val_bpb below 0.991835 while retaining at least 475M trained tokens.

INTENDED_EDIT: Restore the proven fixed two-lag embedding FIR and four handoff gates, then use the untested midpoint between the 768/1024/1280 and 640/1024/1408 short-window schedules.

EVIDENCE: The 768/1024/1280 schedule achieved the best val_bpb of 0.991835, while nearby 640/1024/1408 reached 0.991994 and more distant schedules regressed; their compute-neutral midpoint provides the most informative local refinement.

<<<<<<< SEARCH
        # Channel-wise coefficients for embeddings one and two tokens back,
        # with token-conditioned selection of the useful local-context lag.
        self.embedding_context = nn.Parameter(torch.zeros(2, config.n_embd))
        self.embedding_context_gate = nn.Linear(32, 2, bias=False)
=======
        # Channel-wise coefficients for embeddings one and two tokens back.
        self.embedding_context = nn.Parameter(torch.zeros(2, config.n_embd))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.embedding_context.zero_()
        torch.nn.init.zeros_(self.embedding_context_gate.weight)
=======
        self.embedding_context.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
=======
        long_window = config.sequence_len
        assert pattern == "SSSL"
        # Preserve total short-attention FLOPs while refining the interval
        # between the two strongest staged-window schedules.
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        context_mix = self.embedding_context.numel() + self.embedding_context_gate.weight.numel()
=======
        context_mix = self.embedding_context.numel()
>>>>>>> REPLACE

<<<<<<< SEARCH
        context_params = [self.embedding_context, self.embedding_context_gate.weight]
=======
        context_params = [self.embedding_context]
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Token-conditioned depthwise causal FIR supplies selected local context.
        # Left padding prevents information crossing independent sequence boundaries.
        padded_x = F.pad(x, (0, 0, 2, 0))
        previous_x = padded_x[:, 1:-1]
        previous2_x = padded_x[:, :-2]
        context_mix = torch.tanh(self.embedding_context).to(dtype=x.dtype)
        context_gate = 2 * torch.sigmoid(self.embedding_context_gate(x[..., :32]))
        x = (x +
             context_gate[..., 0, None] * context_mix[0] * previous_x +
             context_gate[..., 1, None] * context_mix[1] * previous2_x)
=======
        # Learned depthwise causal FIR supplies exact local context before attention.
        # Left padding prevents information crossing independent sequence boundaries.
        padded_x = F.pad(x, (0, 0, 2, 0))
        previous_x = padded_x[:, 1:-1]
        previous2_x = padded_x[:, :-2]
        context_mix = torch.tanh(self.embedding_context).to(dtype=x.dtype)
        x = x + context_mix[0] * previous_x + context_mix[1] * previous2_x
>>>>>>> REPLACE