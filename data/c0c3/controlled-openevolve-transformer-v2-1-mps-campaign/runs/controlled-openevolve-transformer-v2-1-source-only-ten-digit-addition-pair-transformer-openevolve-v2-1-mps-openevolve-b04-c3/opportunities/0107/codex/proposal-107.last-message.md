MECHANISM: Cached Fourier positional coefficient pruning

HYPOTHESIS: Fixing the ninth trailing positional-synthesis coefficient at zero will reduce the qualified model to 1,137 parameters while retaining at least 99% accuracy, and caching the input-independent Fourier features will allow all 45,000 updates to finish within the verification limit.

INTENDED_EDIT: Extend the positional-map constraint from eight to nine trailing zeros while preserving full constructor RNG consumption, and precompute the unchanged Fourier feature table once instead of rebuilding it during every forward pass.

EVIDENCE: The 1,138-parameter factorized model achieved 99.78% accuracy; the previous ninth-positional-coefficient probe timed out without an accuracy failure, so repeating that unresolved one-parameter reduction with cheaper, mathematically equivalent positional evaluation is the most direct next test.

<<<<<<< SEARCH
        # Generic Fourier coordinates are synthesized by the qualified dense
        # map with its final output row fixed at zero.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 8)
        )
        self.pos_mix.fixed_weight_trim = (0, 8)
        pos_inv_freq = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        self.register_buffer("pos_inv_freq", pos_inv_freq, persistent=False)
=======
        # Generic Fourier coordinates are synthesized by the qualified dense
        # map with its final output row and one preceding scalar fixed at zero.
        self.pos_mix = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.pos_mix.weight = nn.Parameter(
            torch.empty(cfg.d_model * cfg.d_model - 9)
        )
        self.pos_mix.fixed_weight_trim = (0, 9)
        pos_inv_freq = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        positions = torch.arange(cfg.max_seq_len, dtype=torch.float32)
        angles = positions.unsqueeze(-1) * pos_inv_freq
        pos_features = torch.stack(
            (angles.sin(), angles.cos()),
            dim=-1,
        ).flatten(-2)
        self.register_buffer("pos_features", pos_features, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_weight = F.pad(self.token_emb.weight, (0, 4)).view(
=======
        token_weight = F.pad(self.token_emb.weight, (0, 4)).view(
>>>>>>> REPLACE

<<<<<<< SEARCH
        angles = pos.to(self.pos_inv_freq.dtype).unsqueeze(-1) * self.pos_inv_freq
        pos_features = torch.stack(
            (angles.sin(), angles.cos()),
            dim=-1,
        ).flatten(-2)
        pos_weight = F.pad(self.pos_mix.weight, (0, 8)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
        pos_x = F.linear(pos_features.to(token_x.dtype), pos_weight)
=======
        pos_weight = F.pad(self.pos_mix.weight, (0, 9)).view(
            self.cfg.d_model,
            self.cfg.d_model,
        )
        pos_x = F.linear(
            self.pos_features[:seqlen].to(token_x.dtype),
            pos_weight,
        ).unsqueeze(0)
>>>>>>> REPLACE