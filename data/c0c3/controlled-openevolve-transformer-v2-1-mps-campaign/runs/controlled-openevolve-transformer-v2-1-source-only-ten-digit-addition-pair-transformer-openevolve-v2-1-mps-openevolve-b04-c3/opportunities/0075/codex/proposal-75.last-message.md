MECHANISM: Four-entry positional-mixer constraint

HYPOTHESIS: Fixing a fourth scalar in the qualified dense Fourier positional synthesizer will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,216 to 1,215 learned parameters.

INTENDED_EDIT: Replace the absolute-position embedding with the qualified Fourier synthesizer, learn 60 of its 64 mixer weights, and fix the final four weights at zero while preserving full-matrix initialization RNG consumption.

EVIDENCE: Reference Design 3 achieved 100% accuracy with three fixed positional-mixer scalars and 1,216 parameters; the preceding one- and two-scalar constraints also achieved 100%, motivating the smallest incremental fourth constraint while retaining dense cross-channel mixing.

<<<<<<< SEARCH
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
=======
        self.qkv.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 7))
        self.qkv.fixed_weight_trim = (7, 0)
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)

        # Generic Fourier coordinates are synthesized by a shared dense map.
        # Its final four scalars are fixed at zero while all other channel
        # couplings remain learned. Constructing the full Linear preserves RNG.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 4)
        )
        self.pos_mix.fixed_weight_trim = (0, 4)
        pos_inv_freq = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        self.register_buffer("pos_inv_freq", pos_inv_freq, persistent=False)

        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
                with torch.no_grad():
                    module.weight.copy_(full_weight[7:])
=======
                start, end_trim = module.fixed_weight_trim
                end = full_weight.numel() - end_trim
                with torch.no_grad():
                    module.weight.copy_(full_weight[start:end])
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_scale * token_x + self.pos_emb(pos)
        x = self.drop(x)
=======
        angles = pos.to(self.pos_inv_freq.dtype).unsqueeze(-1) * self.pos_inv_freq
        pos_features = torch.stack(
            (angles.sin(), angles.cos()),
            dim=-1,
        ).flatten(-2)
        pos_weight = F.pad(self.pos_mix.weight, (0, 4)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
        pos_x = F.linear(pos_features.to(token_x.dtype), pos_weight)
        x = self.token_scale * token_x + pos_x
        x = self.drop(x)
>>>>>>> REPLACE