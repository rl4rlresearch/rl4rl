MECHANISM: Fixed-spacing phase sharing with tied terminal lag logits

HYPOTHESIS: Fixing the already-qualified head phase and temperature and tying the final two shared-kernel lag logits will reduce the model from 1,258 to 1,255 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Use the qualified fixed half-period head spacing and common temperature, then constrain the two terminal cyclic-kernel coordinates to share the gauge-fixed value.

EVIDENCE: Fixed-spacing phase-shared attention achieved 99.94% accuracy with 1,256 parameters; its remaining unrestricted 29-coordinate lag kernel has substantial accuracy margin, motivating a one-coordinate terminal-lag tie as the next boundary test.

<<<<<<< SEARCH
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
=======
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the two terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 2)
        )
        self.register_buffer(
            "relative_shift",
            torch.arange(1, n_head, dtype=torch.float32)
            * (max_seq_len / n_head),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(1),
            )
        )
=======
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        head_shift = torch.cat(
            (
                self.relative_shift.new_zeros(1),
                self.relative_shift,
            )
        )
        phase = torch.exp(
>>>>>>> REPLACE

<<<<<<< SEARCH
        lag_bias = torch.fft.irfft(
            torch.fft.rfft(base_bias).unsqueeze(0) * phase,
            n=base_bias.numel(),
            dim=-1,
        )
        lag_bias = lag_bias * head_log_scale.exp()[:, None]
        lag_bias = lag_bias - lag_bias[:, -1:]
=======
        lag_bias = torch.fft.irfft(
            torch.fft.rfft(base_bias).unsqueeze(0) * phase,
            n=base_bias.numel(),
            dim=-1,
        )
        lag_bias = lag_bias - lag_bias[:, -1:]
>>>>>>> REPLACE