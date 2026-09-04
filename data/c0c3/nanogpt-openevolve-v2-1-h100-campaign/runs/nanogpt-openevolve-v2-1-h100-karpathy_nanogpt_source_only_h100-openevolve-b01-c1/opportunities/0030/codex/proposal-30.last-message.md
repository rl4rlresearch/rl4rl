MECHANISM: Head-wise continuous global context

HYPOTHESIS: Giving one attention head full context in every layer will lower val_bpb below 0.984467 by enabling continuous global information exchange while preserving the best design’s exact global/local attention-head compute budget.

INTENDED_EDIT: Replace periodic full-width global layers with three local heads and one global head per layer; parameter count and theoretical attention FLOPs remain unchanged, but context scale becomes a learned channel-wise representation rather than a layer-wise choice.

EVIDENCE: Two full-context layers improved val_bpb to 0.984467, while three regressed to 0.985972. This suggests the load-bearing factor is a limited amount of global computation, challenging the old assumption that it must occur as occasional all-channel consolidation.

<<<<<<< SEARCH
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
=======
    n_kv_head: int = 6
    n_embd: int = 768
    local_window_divisor: int = 8
    n_global_head: int = 1
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
=======
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_global_head = config.n_global_head
        self.n_local_head = self.n_head - self.n_global_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head == self.n_head
        assert 0 < self.n_global_head < self.n_head
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y.contiguous().view(B, T, -1)
=======
        # Heterogeneous context channels: local heads preserve high-resolution
        # nearby structure while a global head communicates across the entire
        # prefix at every depth. c_proj mixes both scales after each attention.
        split = self.n_local_head
        y_local = fa3.flash_attn_func(
            q[:, :, :split], k[:, :, :split], v[:, :, :split],
            causal=True, window_size=window_size,
        )
        y_global = fa3.flash_attn_func(
            q[:, :, split:], k[:, :, split:], v[:, :, split:],
            causal=True, window_size=(T, 0),
        )
        y = torch.cat((y_local, y_global), dim=2).contiguous().view(B, T, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 8
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes
=======
    def _compute_window_sizes(self, config):
        assert config.local_window_divisor > 0
        short_window = config.sequence_len // config.local_window_divisor
        return [(short_window, 0) for _ in range(config.n_layer)]
>>>>>>> REPLACE

<<<<<<< SEARCH
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
=======
        attn_flops = 0
        global_h = self.config.n_global_head
        local_h = h - global_h
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * q * (local_h * effective_seq + global_h * t)
>>>>>>> REPLACE

<<<<<<< SEARCH
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # periodic full-context consolidation at layers 4 and 8
=======
ASPECT_RATIO = 64        # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128           # target head dimension for attention
LOCAL_WINDOW_DIVISOR = 8 # local heads attend over 1/8 of the sequence
N_GLOBAL_HEAD = 1        # one full-context channel at every layer
>>>>>>> REPLACE

<<<<<<< SEARCH
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
=======
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        local_window_divisor=LOCAL_WINDOW_DIVISOR,
        n_global_head=N_GLOBAL_HEAD,
>>>>>>> REPLACE