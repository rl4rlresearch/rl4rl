MECHANISM: Five-entry positional-mixer constraint

HYPOTHESIS: Fixing a fifth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,215 to 1,214 learned parameters.

INTENDED_EDIT: Adopt the qualified seven-entry query-weight gauge and Fourier positional backbone, then learn 59 of 64 positional-mixer weights with the final five fixed at zero while preserving full-matrix initialization RNG consumption.

EVIDENCE: The four-entry positional-mixer design achieved 100% accuracy with 1,215 parameters, following 100% results for one through three fixed entries; this motivates the smallest incremental fifth constraint while retaining dense cross-channel mixing.

<<<<<<< SEARCH
        # Key and value biases are redundant. Q/K basis changes permit four
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 4))
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
=======
        # Key and value biases are redundant. Q/K basis changes permit seven
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
        self.qkv.fixed_weight_trim = (7, 0)
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_weight = F.pad(self.qkv.weight, (4, 0)).view(3 * d_model, d_model)
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
        # Its final five scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 5)
        )
        self.pos_mix.fixed_weight_trim = (0, 5)
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
                    module.weight.copy_(full_weight[4:])
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
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        x = self.token_scale * token_x + self.pos_emb(pos)
        x = self.drop(x)
=======
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        angles = pos.to(self.pos_inv_freq.dtype).unsqueeze(-1) * self.pos_inv_freq
        pos_features = torch.stack(
            (angles.sin(), angles.cos()),
            dim=-1,
        ).flatten(-2)
        pos_weight = F.pad(self.pos_mix.weight, (0, 5)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
        pos_x = F.linear(pos_features.to(token_x.dtype), pos_weight)
        x = self.token_scale * token_x + pos_x
        x = self.drop(x)
>>>>>>> REPLACE