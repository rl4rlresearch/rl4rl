MECHANISM: Positional-scale/query-bias gauge tying

HYPOTHESIS: Tying the learned sinusoidal gain to the query-bias coordinate already shared across heads will preserve adaptive positional scaling and achieve at least 99% accuracy after 75,000 steps with 1,395 parameters.

INTENDED_EDIT: Share one query-bias coordinate across attention heads, reuse that scalar as an additive positional gain offset, remove the separate positional-scale parameter, and validate only at the final positive step.

EVIDENCE: Cross-head sharing achieved 99.97% accuracy with 1,396 parameters, while fixing the positional gain produced 0% accuracy; reusing the shared adaptive scalar targets 1,395 parameters without imposing the failed fixed-gain constraint.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent per-head Q/K basis changes let
        # one query coordinate be shared while keeping both heads adaptive.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
=======
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:1],
                self.qkv.bias[self.head_dim :],
            )
        )
        qkv_bias = torch.cat(
            (
                query_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # A generic Fourier basis replaces the independently learned vector at
        # every position. Attention learns how to use these positional features,
        # while one gain adapts their magnitude relative to token embeddings.
=======
        # A generic Fourier basis replaces the independently learned vector at
        # every position. Its gain reuses the shared query-bias scalar: Q/K
        # basis freedom makes that coordinate a suitable adaptive gauge.
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer("pos_encoding", 0.02 * pos_encoding, persistent=False)
        self.pos_scale = nn.Parameter(torch.ones(()))
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.register_buffer("pos_encoding", 0.02 * pos_encoding, persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
=======
        pos_scale = 1.0 + self.blocks[0].attn.qkv.bias[0]
        x = self.token_emb(idx) + pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if (step > 0 and step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=75000)
>>>>>>> REPLACE