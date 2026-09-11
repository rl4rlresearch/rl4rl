MECHANISM: Nonlinear full-rank tied token manifold

HYPOTHESIS: A five-parameter-per-token angular-product lift will produce a generically full-rank eight-dimensional tied token code and achieve at least 99% accuracy after 45,000 steps with 1,233 learned parameters.

INTENDED_EDIT: Replace the load-bearing assumption that token representations must occupy a learned linear subspace with a nonlinear homogeneous lift from five learned coordinates to all eight residual channels, while restoring the qualified learned positions and shared three-parameter query bias.

EVIDENCE: The rank-six linear token model reached 100%, whereas rank-five zero-padded tokens reached only 54.53%; that test simultaneously reduced learned coordinates and limited token injection and classification to rank five. The nonlinear lift isolates learned degrees of freedom from representational rank while retaining the qualified 1,347-parameter backbone.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant, while value bias is absorbed exactly
        # by the affine output projection. Constructing the original Linear
        # first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Independent Q/K basis changes permit one
        # three-coordinate query bias to be shared by both heads, with its
        # fourth coordinate fixed at zero.
        self.qkv.bias = nn.Parameter(torch.empty(self.head_dim - 1))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
=======
        shared_query_bias = F.pad(self.qkv.bias, (0, 1))
        query_bias = shared_query_bias.repeat(self.n_head)
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

        # Each token learns five coordinates. Three scale-preserving angular
        # product features lift that code nonlinearly into every residual
        # channel, allowing the tied code matrix to have full linear rank
        # without learning eight independent values per token.
        self.token_dim = cfg.d_model - 3
        if self.token_dim < 5:
            raise ValueError("d_model must be at least 8")
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)

        self.apply(self._init_weights)

    def _token_code(self) -> torch.Tensor:
        z = self.token_emb.weight
        a = z[:, [0, 2, 4]]
        b = z[:, [1, 3, 0]]
        angular_products = 2.0 * a * b * torch.rsqrt(
            a.square() + b.square() + 1e-8
        )
        return torch.cat((z, angular_products), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        token_code = self._token_code()
        x = F.embedding(idx, token_code) + self.pos_emb(pos)
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, token_code)
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

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=75000)
>>>>>>> REPLACE