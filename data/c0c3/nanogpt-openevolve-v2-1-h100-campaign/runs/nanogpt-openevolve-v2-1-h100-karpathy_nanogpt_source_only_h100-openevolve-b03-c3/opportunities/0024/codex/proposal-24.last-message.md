MECHANISM: Three-lag causal embedding mixing with pre-long-context attention gating

HYPOTHESIS: Extending the best two-lag embedding FIR to three learned lags will lower val_bpb below 0.992110 while preserving at least 470M trained tokens.

INTENDED_EDIT: Restore the best four-layer handoff-gated design, add a neutral-initialized third causal embedding lag, and optimize all three channel-wise coefficients with AdamW.

EVIDENCE: Two-lag embedding mixing with gates on full-context layers and their immediate predecessors achieved the best result, 0.992110 on 477.1M tokens; the convolution alone also improved to 0.993485, motivating a minimal receptive-field extension without adding per-layer computation.

<<<<<<< SEARCH
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False)
=======
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        pattern = config.window_pattern.upper()
        pattern_idx = layer_idx % len(pattern)
        is_long_layer = pattern[pattern_idx] == "L" or layer_idx == config.n_layer - 1
        precedes_long_layer = pattern[(pattern_idx + 1) % len(pattern)] == "L"
        is_gated_layer = is_long_layer or precedes_long_layer
        self.out_gate = nn.Linear(self.ve_gate_channels, self.n_head, bias=False) if is_gated_layer else None
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Compute value and output controls together on value-embedding layers.
        gate_input = x[..., :self.ve_gate_channels]
        if ve is not None:
            gate_logits = F.linear(
                gate_input,
                torch.cat((self.ve_gate.weight, self.out_gate.weight), dim=0),
            )
            ve_gate_logits = gate_logits[..., :self.n_kv_head]
            out_gate_logits = gate_logits[..., self.n_kv_head:]
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            v = v + (2 * torch.sigmoid(ve_gate_logits)).unsqueeze(-1) * ve
        else:
            out_gate_logits = self.out_gate(gate_input)
=======
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        y = y * (2 * torch.sigmoid(out_gate_logits)).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
=======
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        if self.out_gate is not None:
            out_gate = 2 * torch.sigmoid(self.out_gate(x[..., :self.ve_gate_channels]))
            y = y * out_gate.unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings
=======
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Channel-wise coefficients for embeddings one, two, and three tokens back.
        self.embedding_context = nn.Parameter(torch.zeros(3, config.n_embd))
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
=======
        x = self.transformer.wte(idx)
        # Learned depthwise causal FIR supplies exact local context before attention.
        # Left padding prevents information crossing independent sequence boundaries.
        padded_x = F.pad(x, (0, 0, 3, 0))
        previous_x = padded_x[:, 2:-1]
        previous2_x = padded_x[:, 1:-2]
        previous3_x = padded_x[:, :-3]
        context_mix = torch.tanh(self.embedding_context).to(dtype=x.dtype)
        x = (x + context_mix[0] * previous_x +
             context_mix[1] * previous2_x +
             context_mix[2] * previous3_x)
        x = norm(x)
>>>>>>> REPLACE