MECHANISM: Long-context-selective gating with causal embedding mixing

HYPOTHESIS: Applying the learned two-lag embedding convolution while retaining output gates only in full-context layers will lower val_bpb below 0.992628 and restore training volume to roughly 480M tokens or more.

INTENDED_EDIT: Add the proven channel-wise two-lag causal embedding convolution and remove attention-output gates from the six short-context layers.

EVIDENCE: The full convolution-plus-gating design achieved the best val_bpb, 0.992628, but processed only 467.1M tokens; long-context-only gating was just 0.000228 worse than all-layer gating while increasing training volume, suggesting short-layer gates may not justify their fixed-time cost.

<<<<<<< SEARCH
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
=======
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        pattern = config.window_pattern.upper()
        is_long_layer = pattern[layer_idx % len(pattern)] == "L" or layer_idx == config.n_layer - 1
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_long_layer else None
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        out_gate = 2 * torch.sigmoid(self.out_gate(x[..., :self.ve_gate_channels]))
        y = y * out_gate.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
=======
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        if self.out_gate is not None:
            out_gate = 2 * torch.sigmoid(self.out_gate(x[..., :self.ve_gate_channels]))
            y = y * out_gate.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
=======
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Channel-wise coefficients for embeddings one and two tokens back.
        self.embedding_context = nn.Parameter(torch.zeros(2, config.n_embd))
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Value embeddings
=======
        # Residual and causal embedding-mixing parameters
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        self.embedding_context.zero_()
        # Value embeddings
>>>>>>> REPLACE

<<<<<<< SEARCH
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.out_gate.weight)
=======
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            if block.attn.out_gate is not None:
                torch.nn.init.zeros_(block.attn.out_gate.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
=======
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.embedding_context.numel())
>>>>>>> REPLACE

<<<<<<< SEARCH
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }
=======
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        context_mix = self.embedding_context.numel()
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + context_mix + scalars
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'context_mix': context_mix,
            'scalars': scalars, 'total': total,
        }
>>>>>>> REPLACE

<<<<<<< SEARCH
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
=======
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        context_params = [self.embedding_context]
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) +
            len(x0_params) + len(context_params))
>>>>>>> REPLACE

<<<<<<< SEARCH
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
=======
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
            dict(kind='adamw', params=context_params, lr=scalar_lr * 0.02, betas=adam_betas, eps=1e-10, weight_decay=0.0),
        ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
=======
        x = self.transformer.wte(idx)
        # Learned depthwise causal FIR supplies exact local context before attention.
        # Left padding prevents information crossing independent sequence boundaries.
        padded_x = F.pad(x, (0, 0, 2, 0))
        previous_x = padded_x[:, 1:-1]
        previous2_x = padded_x[:, :-2]
        context_mix = torch.tanh(self.embedding_context).to(dtype=x.dtype)
        x = x + context_mix[0] * previous_x + context_mix[1] * previous2_x
        x = norm(x)
        x0 = x
>>>>>>> REPLACE