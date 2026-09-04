MECHANISM: Incremental single-coordinate query-bias gauge fixing with learned-scale sinusoidal positions

HYPOTHESIS: Fixing one query-bias coordinate while retaining the qualified learned positional gain will achieve at least 99% accuracy with 1,396 parameters after 75,000 steps.

INTENDED_EDIT: Adopt query-only QKV bias and learned-scale sinusoidal positions, remove one query-bias coordinate through an exact per-head rotational gauge, train for 75,000 steps, and reduce validation frequency to remain within the verification time limit.

EVIDENCE: The 1,397-parameter sinusoidal design achieved 99.69% accuracy at 74,999 steps, whereas removing six query-bias coordinates at once collapsed to 15.31%; removing only one coordinate is the closest informative intermediate test. Recent 75,000-step timeouts motivate reducing validation overhead without changing optimization updates.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # All key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key and value biases are functionally redundant. One query-bias
        # coordinate is fixed by the joint per-head Q/K rotational gauge.
        # Constructing the original Linear preserves the constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(d_model),
                self.qkv.bias[d_model:],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model + 1),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        positions = torch.arange(cfg.max_seq_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, cfg.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / cfg.d_model)
        )
        pos_encoding = torch.zeros(cfg.max_seq_len, cfg.d_model)
        pos_encoding[:, 0::2] = torch.sin(positions * frequencies)
        pos_encoding[:, 1::2] = torch.cos(
            positions * frequencies[: pos_encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("pos_encoding", 0.02 * pos_encoding, persistent=False)
        self.pos_scale = nn.Parameter(torch.ones(()))
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
=======
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=45000)
=======
    p.add_argument("--train-steps", type=int, default=75000)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=1000)
>>>>>>> REPLACE