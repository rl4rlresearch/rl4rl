MECHANISM: Rotary relative-position attention with affine Q/K routing

HYPOTHESIS: Replacing learned additive position vectors with parameter-free rotary Q/K phases, while retaining the qualified one-coordinate query sharing, will achieve at least 99% accuracy after 45,000 steps with 1,395 learned parameters.

INTENDED_EDIT: Replace the assumption that positions require residual-stream embeddings with relative rotary attention; repurpose the former value-bias parameters as useful key biases, eliminate value bias, share one query coordinate, and use positive-step endpoint validation.

EVIDENCE: Fixed sinusoidal positions reached 99.69%, proving a learned position table is unnecessary; one-coordinate query sharing reached 99.97%; and the current 45,000-step schedule completed at 99.92%. Rotary phases provide a different, direct positional-routing mechanism without the sinusoidal model’s learned gain.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # All key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        if self.head_dim % 2 != 0:
            raise ValueError("rotary attention requires an even head dimension")

        self.qkv = nn.Linear(d_model, 3 * d_model)
        # RoPE makes a key bias position-dependent, so retain affine Q/K
        # routing. Value bias remains exactly absorbable by the output bias.
        # One corresponding query coordinate is shared across the first two
        # heads, preserving the qualified one-parameter quotient.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model - 1))
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        rope_inv_freq = torch.exp(
            torch.arange(0, self.head_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.head_dim)
        )
        self.register_buffer("rope_inv_freq", rope_inv_freq, persistent=False)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv_bias = torch.cat(
            (
                self.qkv.bias[:d_model],
                self.qkv.bias.new_zeros(d_model),
                self.qkv.bias[d_model:],
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        bsz, seqlen, d_model = x.shape
        query_params = self.qkv.bias[: d_model - 1]
        query_bias = torch.cat(
            (
                query_params[: self.head_dim],
                query_params[:1],
                query_params[self.head_dim :],
            )
        )
        key_bias = self.qkv.bias[d_model - 1 :]
        qkv_bias = torch.cat(
            (
                query_bias,
                key_bias,
                self.qkv.bias.new_zeros(d_model),
            )
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        positions = torch.arange(
            seqlen, device=x.device, dtype=self.rope_inv_freq.dtype
        )
        angles = positions.unsqueeze(1) * self.rope_inv_freq.unsqueeze(0)
        cos = angles.cos().to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
        sin = angles.sin().to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)

        q_even, q_odd = q[..., 0::2], q[..., 1::2]
        k_even, k_odd = k[..., 0::2], k[..., 1::2]
        q = torch.stack(
            (q_even * cos - q_odd * sin, q_even * sin + q_odd * cos), dim=-1
        ).flatten(-2)
        k = torch.stack(
            (k_even * cos - k_odd * sin, k_even * sin + k_odd * cos), dim=-1
        ).flatten(-2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        # Position enters through rotary phases inside causal attention rather
        # than through a learned additive residual-stream representation.
        x = self.drop(self.token_emb(idx))
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if (step > 0 and step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=45000)
>>>>>>> REPLACE