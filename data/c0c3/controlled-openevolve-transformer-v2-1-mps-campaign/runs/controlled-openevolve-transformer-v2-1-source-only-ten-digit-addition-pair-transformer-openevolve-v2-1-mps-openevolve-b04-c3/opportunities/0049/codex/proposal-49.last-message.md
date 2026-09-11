MECHANISM: Canonical fixed query gauge with channel-adaptive Fourier positions

HYPOTHESIS: Fixing each head’s query bias to a nonzero canonical direction and reallocating six of the seven removed parameters to per-channel positional gains will reach at least 99% accuracy with 1,395 parameters after 45,000 steps.

INTENDED_EDIT: Remove learned query biases using the exact per-head Q/K basis-change symmetry, add seven learned sinusoidal channel gains with one fixed gain, and use the completed-duration 45,000-step schedule.

EVIDENCE: The 1,396-parameter shared-query model achieved 99.97%, while prior 1,395-parameter constraints retained the poorly conditioned single positional gain and failed or timed out. Reference Design 3 reached 99.92% in 45,000 steps with more adaptable positional representations, motivating reallocating redundant query coordinates to positional conditioning.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. One query coordinate is shared across the
        # first two heads; independent Q/K rotations make that equality a
        # one-dimensional gauge choice while keeping every coordinate adaptive.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key and value biases are redundant. A nonzero query-bias vector can
        # be mapped to a fixed canonical direction by an invertible Q/K basis
        # change within each head, so its learned coordinates are also gauge.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = None
        fixed_query_bias = torch.zeros(d_model)
        fixed_query_bias[:: self.head_dim] = 1.0
        self.register_buffer("fixed_query_bias", fixed_query_bias, persistent=False)
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        qkv_bias = torch.cat(
            (
                self.fixed_query_bias,
                x.new_zeros(2 * d_model),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_scale = nn.Parameter(torch.ones(()))
=======
        # Seven adaptive channel gains improve the conditioning of the fixed
        # Fourier basis; the final gain is fixed to remove its scale gauge.
        self.pos_scale = nn.Parameter(torch.ones(cfg.d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
=======
        position_scale = torch.cat(
            (self.pos_scale, self.pos_scale.new_ones(1))
        )
        x = self.token_emb(idx) + position_scale * self.pos_encoding[:seqlen].unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=75000)
=======
    p.add_argument("--train-steps", type=int, default=45000)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=75000)
=======
    p.add_argument("--eval-interval", type=int, default=45000)
>>>>>>> REPLACE