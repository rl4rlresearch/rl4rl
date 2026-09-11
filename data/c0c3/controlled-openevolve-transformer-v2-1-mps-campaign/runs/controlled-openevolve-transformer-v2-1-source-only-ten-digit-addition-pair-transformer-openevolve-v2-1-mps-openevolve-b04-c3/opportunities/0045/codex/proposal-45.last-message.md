MECHANISM: Variance-matched multiscale Fourier positions with two-coordinate query-bias sharing

HYPOTHESIS: A better-conditioned dyadic Fourier basis will let the two-coordinate-shared model reach at least 99% accuracy in 45,000 updates with 1,395 parameters.

INTENDED_EDIT: Share two corresponding query-bias coordinates, replace poorly distributed sinusoidal frequencies with variance-matched dyadic Fourier features, and perform validation only at the final positive step.

EVIDENCE: One-coordinate sharing reached 99.97% with 1,396 parameters, while Reference Design 3 reached 99.92% in 45,000 steps with well-conditioned learned positions; the old sinusoidal two-coordinate model’s 11.6% result motivates improving positional conditioning instead of further increasing its learning rate.

<<<<<<< SEARCH
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. One query coordinate is shared across the
        # first two heads; independent Q/K rotations make that equality a
        # one-dimensional gauge choice while keeping every coordinate adaptive.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        # Key bias is softmax-invariant and value bias is absorbed by the
        # affine output projection. Two corresponding query coordinates are
        # shared across the first two heads.
        # Constructing the original Linear first preserves its RNG consumption.
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:1],
                self.qkv.bias[self.head_dim :],
            )
        )
=======
        query_bias = torch.cat(
            (
                self.qkv.bias[: self.head_dim],
                self.qkv.bias[:2],
                self.qkv.bias[self.head_dim :],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        # Dyadic Fourier harmonics distribute the available dimensions across
        # the sequence's useful frequency range. Scaling by sqrt(2) gives each
        # position the same 0.02 RMS as a freshly initialized embedding vector.
        positions = torch.arange(cfg.max_seq_len, dtype=torch.float32).unsqueeze(1)
        harmonics = 2.0 ** torch.arange(0, cfg.d_model // 2, dtype=torch.float32)
        frequencies = (2.0 * math.pi / cfg.max_seq_len) * harmonics
        pos_encoding = torch.zeros(cfg.max_seq_len, cfg.d_model)
        pos_encoding[:, 0::2] = torch.sin(positions * frequencies)
        pos_encoding[:, 1::2] = torch.cos(
            positions * frequencies[: pos_encoding[:, 1::2].shape[1]]
        )
        self.register_buffer(
            "pos_encoding",
            math.sqrt(2.0) * 0.02 * pos_encoding,
            persistent=False,
        )
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

<<<<<<< SEARCH
        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        if ((step + 1) % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE