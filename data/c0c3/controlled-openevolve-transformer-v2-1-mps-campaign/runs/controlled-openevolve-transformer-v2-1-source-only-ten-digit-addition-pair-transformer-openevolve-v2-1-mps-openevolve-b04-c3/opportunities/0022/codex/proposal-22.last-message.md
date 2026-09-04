MECHANISM: Per-head query-bias gauge fixing with learned-scale sinusoidal positions

HYPOTHESIS: Because each attention head’s Q/K coordinates can be jointly rotated without changing attention logits, retaining one learned query-bias coordinate per head will preserve at least 99% accuracy after 75,000 steps while reducing the qualified 1,397-parameter sinusoidal design to 1,391 parameters.

INTENDED_EDIT: Adopt the qualified learned-scale sinusoidal positional representation and exact key/value-bias quotient, represent each four-dimensional query bias with one learned coordinate, and train for 75,000 steps.

EVIDENCE: The 1,397-parameter learned-scale sinusoidal design achieved 99.69% accuracy at 74,999 steps. This patch preserves its learned positional gain and removes only six within-head query-bias orientation degrees that can be absorbed by jointly rotating the learned query and key projections.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Seven key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(3 * d_model - 7))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Jointly rotating Q and K within a head leaves their dot products
        # unchanged, so one query-bias coordinate represents its magnitude.
        # Key bias is softmax-invariant and value bias is absorbed by proj.bias.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(n_head))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(7),
                self.qkv.bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        q_bias = F.pad(
            self.qkv.bias.unsqueeze(1), (0, self.head_dim - 1)
        ).reshape(-1)
        qkv_bias = torch.cat(
            (
                q_bias,
                self.qkv.bias.new_zeros(2 * d_model),
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # A generic Fourier basis replaces independently learned position
        # vectors while retaining one learned magnitude.
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
        x = self.drop(x)
=======
        x = self.token_emb(idx) + self.pos_scale * self.pos_encoding[:seqlen].unsqueeze(0)
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=40000)
=======
    p.add_argument("--train-steps", type=int, default=75000)
>>>>>>> REPLACE