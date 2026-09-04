MECHANISM: Q/K scale-gauge positional gain

HYPOTHESIS: Encoding the sinusoidal position gain in the exact reciprocal Q/K scaling gauge will retain at least 99% accuracy after 75,000 steps with 1,396 parameters.

INTENDED_EDIT: Adopt the qualified query-only QKV bias and sinusoidal positions, derive their gain from the learned Q/K weight-energy ratio instead of a separate parameter, and use sparse validation to reduce verification time.

EVIDENCE: The learned-gain sinusoidal design achieved 99.69% accuracy with 1,397 parameters at 74,999 steps. Fixed-gain variants produced no accuracy evidence because they timed out, while aggressive query-bias gauge fixing collapsed accuracy; recycling the smooth global Q/K scaling symmetry preserves adaptive positional amplitude without fixing query-bias coordinates.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Seven key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 7))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(7),
                self.qkv.bias[d_model:],
            )
        )
=======
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
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
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        qkv_weight = self.blocks[0].attn.qkv.weight
        d_model = self.cfg.d_model
        q_energy = qkv_weight[:d_model].square().mean()
        k_energy = qkv_weight[d_model : 2 * d_model].square().mean()
        pos_scale = (q_energy / k_energy).pow(0.25)
        x = self.token_emb(idx) + pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=40000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--train-steps", type=int, default=75000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-interval", type=int, default=15000)
>>>>>>> REPLACE