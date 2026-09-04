MECHANISM: Global lexical-phase gauge fixing

HYPOTHESIS: Fixing one token’s phase to zero will reduce the verified model from 606 to 605 parameters while retaining at least 99% accuracy, because a common phase shift is an exact rotational gauge of the four-dimensional harmonic lexical representation.

INTENDED_EDIT: Remove one learned token-phase coordinate and rotate the lexical projection and bias at initialization to preserve the model function exactly.

EVIDENCE: The 606-parameter model achieved 99.89% accuracy, while removing the final LayerNorm gain failed catastrophically; this instead removes a mathematically redundant coordinate without reducing the learned lexical function class.

<<<<<<< SEARCH
            token_phase = torch.atan2(
                initialized_token_code[:, 0],
                initialized_token_code[:, 1],
            )

        self.token_phase = nn.Parameter(token_phase.clone())
        self.token_radius = nn.Parameter(token_radius.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
=======
            token_phase = torch.atan2(
                initialized_token_code[:, 0],
                initialized_token_code[:, 1],
            )

            # A common phase offset is only a rotation of the four harmonic
            # coordinates. Fix the first token's phase and absorb that rotation
            # into the learned projection and lexical bias.
            phase_origin = token_phase[0].clone()
            token_phase = token_phase - phase_origin
            phase_rotation = torch.eye(
                token_rank,
                device=token_proj.device,
                dtype=token_proj.dtype,
            )
            phase_cos = torch.cos(phase_origin)
            phase_sin = torch.sin(phase_origin)
            phase_rotation[0, 0] = phase_cos
            phase_rotation[0, 1] = -phase_sin
            phase_rotation[1, 0] = phase_sin
            phase_rotation[1, 1] = phase_cos
            double_phase_cos = torch.cos(2.0 * phase_origin)
            double_phase_sin = torch.sin(2.0 * phase_origin)
            phase_rotation[2, 2] = double_phase_cos
            phase_rotation[2, 3] = -double_phase_sin
            phase_rotation[3, 2] = double_phase_sin
            phase_rotation[3, 3] = double_phase_cos
            token_proj = phase_rotation @ token_proj
            self.ln_f_token_bias.copy_(
                phase_rotation @ self.ln_f_token_bias
            )

        self.token_phase = nn.Parameter(token_phase[1:].clone())
        self.token_radius = nn.Parameter(token_radius.clone())
        self.token_proj = nn.Parameter(token_proj.clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        token_code = self.token_radius.unsqueeze(1) * torch.stack(
            (
                torch.sin(self.token_phase),
                torch.cos(self.token_phase),
                torch.sin(2.0 * self.token_phase),
                torch.cos(2.0 * self.token_phase),
            ),
=======
        token_phase = torch.cat(
            (self.token_phase.new_zeros(1), self.token_phase)
        )
        token_code = self.token_radius.unsqueeze(1) * torch.stack(
            (
                torch.sin(token_phase),
                torch.cos(token_phase),
                torch.sin(2.0 * token_phase),
                torch.cos(2.0 * token_phase),
            ),
>>>>>>> REPLACE