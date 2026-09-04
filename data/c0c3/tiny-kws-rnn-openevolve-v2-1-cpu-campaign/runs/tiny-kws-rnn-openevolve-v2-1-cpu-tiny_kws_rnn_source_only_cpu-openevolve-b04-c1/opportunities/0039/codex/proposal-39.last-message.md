MECHANISM: Orthonormal lossless mel-coordinate elimination

HYPOTHESIS: Mapping normalized 20-band frames into a well-conditioned 19-dimensional Helmert contrast basis will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 3.55M versus the verified 25-step model.

INTENDED_EDIT: Preserve affine input normalization, replace the raw 20 normalized coordinates with 19 orthonormal spectral contrasts, and structurally reduce the GRU input width from 20 to 19.

EVIDENCE: Direct coordinate omission with a 19-input GRU narrowly missed at 84.91%; because normalized frames lie in a 19-dimensional zero-mean subspace, an orthonormal contrast basis preserves that information while avoiding the poor conditioning of reconstructing the omitted coordinate as the sum of the other 19.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)

    def initial_state(
=======
        self.input_norm = nn.LayerNorm(20)
        self.register_buffer(
            "contrast_index",
            torch.arange(1, 20, dtype=torch.float32),
            persistent=False,
        )
        self.gru = nn.GRU(19, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)

    def _spectral_contrasts(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        index = self.contrast_index.to(dtype=normalized.dtype)
        prefix = normalized.cumsum(dim=-1)[..., :-1]
        scale = torch.sqrt(index * (index + 1.0))
        return (prefix - index * normalized[..., 1:]) / scale

    def initial_state(
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self._spectral_contrasts(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self._spectral_contrasts(frames),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE