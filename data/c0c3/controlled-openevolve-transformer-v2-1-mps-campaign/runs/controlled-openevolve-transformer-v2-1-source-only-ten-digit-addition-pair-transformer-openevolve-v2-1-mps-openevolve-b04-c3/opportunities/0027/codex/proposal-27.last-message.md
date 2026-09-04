MECHANISM: Fixed-amplitude sinusoidal positions with endpoint-only validation

HYPOTHESIS: Fixing the sinusoidal amplitude at 0.02 will achieve at least 99% accuracy with 1,396 parameters after 75,000 steps, while endpoint-only validation will allow verification to finish within the time limit.

INTENDED_EDIT: Adopt query-only QKV bias, replace learned positional embeddings with a fixed sinusoidal buffer, remove the learned positional gain, and validate only at the final training step.

EVIDENCE: The learned-gain sinusoidal design achieved 99.69% accuracy with 1,397 parameters at 74,999 steps. Prior fixed-gain runs produced no accuracy evidence solely because they timed out, while endpoint-only validation allowed a separate 75,000-step 1,396-parameter experiment to finish.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # All key-bias coordinates are softmax-invariant. Constructing the
        # original Linear first preserves the baseline constructor RNG stream.
        self.qkv.bias = nn.Parameter(torch.empty(2 * d_model))
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
                self.qkv.bias.new_zeros(d_model),
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
=======
        x = self.token_emb(idx) + self.pos_encoding[:seqlen].unsqueeze(0)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if ((step > 0 and step % train_cfg.eval_interval == 0)
                or (step == train_cfg.train_steps - 1)):
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--train-steps", type=int, default=45000)
=======
    p.add_argument("--train-steps", type=int, default=75000)
>>>>>>> REPLACE

<<<<<<< SEARCH
    p.add_argument("--eval-interval", type=int, default=300)
=======
    p.add_argument("--eval-interval", type=int, default=75000)
>>>>>>> REPLACE