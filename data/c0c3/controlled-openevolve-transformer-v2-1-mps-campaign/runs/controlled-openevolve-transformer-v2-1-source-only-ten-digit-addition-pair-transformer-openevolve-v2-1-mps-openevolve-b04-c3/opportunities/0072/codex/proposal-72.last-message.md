MECHANISM: Single-entry positional-mixer constraint

HYPOTHESIS: Fixing one scalar in the qualified dense Fourier positional synthesizer at zero will retain at least 99% accuracy after 45,000 updates with 1,218 learned parameters.

INTENDED_EDIT: Adopt the qualified seven-entry query-weight gauge and Fourier positional backbone, then learn 63 of the 64 positional-mixer weights while fixing its final entry at zero and preserving full-matrix initialization RNG consumption.

EVIDENCE: The dense 8×8 Fourier synthesizer achieved 100% accuracy with 1,219 parameters, while the aggressive eight-gain restriction failed at 46.49%; retaining 63 cross-channel coefficients tests the smallest informative reduction without repeating that severe loss of positional mixing capacity.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key and value biases are redundant. Q/K basis changes permit three
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 3))
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key and value biases are redundant. Q/K basis changes permit seven
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
        self.qkv.fixed_weight_trim = (7, 0)
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_weight = F.pad(self.qkv.weight, (3, 0)).view(3 * d_model, d_model)
=======
        qkv_weight = F.pad(self.qkv.weight, (7, 0)).view(3 * d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)

        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final scalar is fixed at zero while all other channel couplings
        # remain learned. Constructing the full Linear preserves RNG use.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 1)
        )
        self.pos_mix.fixed_weight_trim = (0, 1)
        pos_inv_freq = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        self.register_buffer("pos_inv_freq", pos_inv_freq, persistent=False)

        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[3:])
=======
            if module.weight.ndim == 1:
                full_weight = module.weight.new_empty(
                    module.out_features * module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                start, end_trim = module.fixed_weight_trim
                end = full_weight.numel() - end_trim
                with torch.no_grad():
                    module.weight.copy_(full_weight[start:end])
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        x = self.token_scale * token_x + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        angles = pos.to(self.pos_inv_freq.dtype).unsqueeze(-1) * self.pos_inv_freq
        pos_features = torch.stack(
            (angles.sin(), angles.cos()),
            dim=-1,
        ).flatten(-2)
        pos_weight = F.pad(self.pos_mix.weight, (0, 1)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
        pos_x = F.linear(pos_features.to(token_x.dtype), pos_weight)
        x = self.token_scale * token_x + pos_x
        x = self.drop(x)
>>>>>>> REPLACE