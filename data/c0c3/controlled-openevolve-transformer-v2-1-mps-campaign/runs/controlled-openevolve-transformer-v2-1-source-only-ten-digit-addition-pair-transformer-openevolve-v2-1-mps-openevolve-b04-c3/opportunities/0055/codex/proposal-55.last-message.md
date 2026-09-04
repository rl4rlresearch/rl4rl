MECHANISM: Rank-seven learned positional subspace

HYPOTHESIS: Preserving the qualified rank-six token code and shared query-bias gauge while reducing learned positions from eight to seven dimensions will retain at least 99% accuracy after 45,000 updates with 1,328 parameters.

INTENDED_EDIT: Adopt the qualified 1,351-parameter learned-position architecture, lift seven-dimensional learned position codes into the eight-dimensional residual stream, and perform validation only at the final training step.

EVIDENCE: Reference Design 3 achieved 100% accuracy with 1,351 parameters, while reducing token rank from six to five fell to 54.53% and sinusoidal positions repeatedly failed or timed out; this preserves both proven components and tests the unexamined redundancy of one positional coordinate.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. One query coordinate is shared across the
        # first two heads using their independent Q/K basis-change freedom.
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
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # A generic Fourier basis replaces the independently learned vector at
        # every position. Attention learns how to use these positional features,
        # while one gain adapts their magnitude relative to token embeddings.
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
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.cfg = cfg

        # Token identities retain the qualified six-dimensional tied code.
        self.token_dim = cfg.d_model - 2
        if self.token_dim <= 0:
            raise ValueError("d_model must be at least 3")
        self.token_scale = math.sqrt(cfg.d_model / self.token_dim)
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)

        # Each position remains independently learned, but in a seven-
        # dimensional subspace lifted into the full residual stream.
        self.pos_dim = cfg.d_model - 1
        self.pos_scale = math.sqrt(cfg.d_model / self.pos_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, self.pos_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)

        self.apply(self._init_weights)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_x = F.pad(
            self.token_emb(idx),
            (0, self.cfg.d_model - self.token_dim),
        )
        pos_x = F.pad(
            self.pos_emb(pos),
            (0, self.cfg.d_model - self.pos_dim),
        )
        x = self.token_scale * token_x + self.pos_scale * pos_x
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(
            x[..., : self.token_dim],
            self.token_scale * self.token_emb.weight,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if step == train_cfg.train_steps - 1:
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=75000)
=======
    p.add_argument("--train-steps", type=int, default=45000)
>>>>>>> REPLACE