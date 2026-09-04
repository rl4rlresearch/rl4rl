MECHANISM: Shared Fourier positional synthesis

HYPOTHESIS: Replacing 98 independent eight-dimensional position vectors with a learned shared projection of generic sinusoidal coordinates will retain at least 99% accuracy after 45,000 updates while removing 720 positional parameters from the qualified seven-entry query-gauge backbone.

INTENDED_EDIT: Adopt the proven seven-zero query-weight constraint, preserve the rank-six tied token code and attention/MLP computation, and replace the learned absolute-position lookup table with a parameter-efficient 8×8 learned Fourier synthesizer.

EVIDENCE: The seven-entry query-weight design achieved 100% accuracy with 1,339 parameters, while the five-coordinate nonlinear token lift failed at 56.64%, identifying token identity geometry as load-bearing. Every qualified design still assumes an independent learned vector for every position; addition instead presents repeated position-to-position routing for which shared sinusoidal coordinates are a plausible structured representation.

<<<<<<< SEARCH
        # Key and value biases are redundant. Q/K basis changes permit five
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 5))
=======
        # Key and value biases are redundant. Q/K basis changes permit seven
        # query-weight entries to be fixed at zero and the query bias to be
        # shared across heads with its final coordinate fixed at zero.
        # Constructing the original Linear first preserves constructor RNG use.
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_weight = F.pad(self.qkv.weight, (5, 0)).view(3 * d_model, d_model)
=======
        qkv_weight = F.pad(self.qkv.weight, (7, 0)).view(3 * d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)

        # Positions are represented by a generic fixed Fourier coordinate
        # system and synthesized by one learned map shared across all steps.
        # This replaces an independent d_model-vector lookup at every position.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        pos_inv_freq = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        self.register_buffer("pos_inv_freq", pos_inv_freq, persistent=False)

        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
                    module.weight.copy_(full_weight[5:])
=======
                    module.weight.copy_(full_weight[7:])
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        x = self.token_scale * token_x + self.pos_emb(pos)
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
        pos_x = self.pos_mix(pos_features.to(token_x.dtype))
        x = self.token_scale * token_x + pos_x
>>>>>>> REPLACE