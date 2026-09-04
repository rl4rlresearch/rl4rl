MECHANISM: Phase-shared discrete lag attention

HYPOTHESIS: A full-resolution learned lag kernel shared through per-head learned cyclic shifts and temperatures will retain at least 99% accuracy with 1,251 parameters, because the heads need distinct routing locations but may not need independently learned kernel shapes.

INTENDED_EDIT: Replace the old assumption that each head requires an unrelated dense lag table with one unrestricted discrete kernel whose learned phase and temperature produce distinct head routes, reducing the attention routing parameters from 60 to 32.

EVIDENCE: Dense stationary lag routing achieved 99.97%, showing discrete lag resolution is load-bearing; the signed-table design achieved only 0.15%, showing opposite logits are too restrictive. Learned translation and scaling preserve discrete resolution while avoiding that failed complementary-sign constraint.

<<<<<<< SEARCH
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )
=======
        # Share one unrestricted discrete kernel while allowing each head to
        # learn a cyclic displacement and softmax temperature.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 1)
        )
        self.relative_shift = nn.Parameter(
            torch.arange(1, n_head, dtype=torch.float32)
            * (max_seq_len / n_head)
        )
        self.relative_log_scale = nn.Parameter(
            torch.zeros(n_head - 1)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        lag_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=-1,
        )
        att = lag_bias[:, lag].unsqueeze(0)
=======
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(1),
            )
        )
        frequency = torch.arange(
            base_bias.numel() // 2 + 1,
            device=x.device,
            dtype=base_bias.dtype,
        )
        head_shift = torch.cat(
            (
                self.relative_shift.new_zeros(1),
                self.relative_shift,
            )
        )
        head_log_scale = torch.cat(
            (
                self.relative_log_scale.new_zeros(1),
                self.relative_log_scale,
            )
        )
        phase = torch.exp(
            -2j
            * math.pi
            * head_shift[:, None]
            * frequency[None, :]
            / base_bias.numel()
        )
        lag_bias = torch.fft.irfft(
            torch.fft.rfft(base_bias).unsqueeze(0) * phase,
            n=base_bias.numel(),
            dim=-1,
        )
        lag_bias = lag_bias * head_log_scale.exp()[:, None]
        lag_bias = lag_bias - lag_bias[:, -1:]
        att = lag_bias[:, lag].unsqueeze(0)
>>>>>>> REPLACE