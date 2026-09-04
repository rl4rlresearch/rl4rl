MECHANISM: Four-way terminal lag-logit tying

HYPOTHESIS: Extending the qualified three-coordinate terminal lag tie to four coordinates will reduce the model from 1,254 to 1,253 learned parameters while retaining at least 99% accuracy, because the remaining shared lag kernel preserves nearly all routing capacity.

INTENDED_EDIT: Replace independent per-head lag tables with the qualified fixed-spacing shared cyclic kernel and constrain its final four logits to the gauge-fixed value.

EVIDENCE: The three-way terminal tie achieved 99.82% accuracy with 1,254 parameters, leaving a 0.82-point margin above the requirement and directly motivating the next isolated one-coordinate tie.

<<<<<<< SEARCH
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
=======
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the four terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 4)
        )
        self.register_buffer(
            "relative_shift",
            torch.arange(1, n_head, dtype=torch.float32)
            * (max_seq_len / n_head),
        )

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
>>>>>>> REPLACE

<<<<<<< SEARCH
        positions = torch.arange(seqlen, device=x.device)
        lag = (positions[:, None] - positions[None, :]).clamp_min(0)
        lag_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=-1,
        )
        att = lag_bias[:, lag].unsqueeze(0)
=======
        positions = torch.arange(seqlen, device=x.device)
        lag = (positions[:, None] - positions[None, :]).clamp_min(0)
        base_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(4),
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
        lag_bias = lag_bias - lag_bias[:, -1:]
        att = lag_bias[:, lag].unsqueeze(0)
>>>>>>> REPLACE