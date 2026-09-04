MECHANISM: Learned low-dimensional token codebook with asymmetric input/output lifts

HYPOTHESIS: A four-dimensional learned token codebook with separate learned projections into and out of the eight-dimensional residual stream will reach at least 99% accuracy after 75,000 steps while reducing the model from 1,397 to 1,005 parameters.

INTENDED_EDIT: Replace the full-width tied token embedding/head with a shared four-dimensional learned codebook and orthogonally initialized input/output projections; retain the qualified attention, MLP, learned sinusoidal gain, and use endpoint-only validation.

EVIDENCE: The 1,397-parameter sinusoidal model reached 99.69%, while attempts to constrain its positional gain or query-bias mechanism failed or approached the threshold. This instead challenges the shared assumption that every vocabulary token needs an unconstrained eight-dimensional interface: the learned transformer still computes in eight dimensions, but token identity and logits pass through a compact learned code space.

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # A generic Fourier basis replaces the independently learned vector at
=======
        self.cfg = cfg

        # Token identities live in a compact learned code space. Independent
        # learned lifts let the residual stream interpret input and output
        # codes differently without storing a full-width vector per token.
        self.token_code_dim = 4
        self.token_emb = nn.Embedding(cfg.vocab_size, self.token_code_dim)
        self.token_in = nn.Linear(self.token_code_dim, cfg.d_model, bias=False)

        # A generic Fourier basis replaces the independently learned vector at
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
=======
        self.token_out = nn.Linear(cfg.d_model, self.token_code_dim, bias=False)

        self.apply(self._init_weights)
        nn.init.normal_(
            self.token_emb.weight,
            mean=0.0,
            std=0.02 * math.sqrt(cfg.d_model / self.token_code_dim),
        )
        nn.init.orthogonal_(self.token_in.weight)
        nn.init.orthogonal_(self.token_out.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
=======
        x = self.token_in(self.token_emb(idx))
        x = x + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(self.token_out(x), self.token_emb.weight)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=75000)
>>>>>>> REPLACE