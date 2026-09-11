MECHANISM: Adjacent high-frequency mel-band fusion

HYPOTHESIS: Fusing the two highest adjacent mel bands will preserve at least 85% validation accuracy while reducing the successful 25-step model by exactly 174 GRU MACs per recurrent step, approximately 3.55M validation MACs overall.

INTENDED_EDIT: Keep the verified 58-unit GRU, 25-frame schedule, and temporal readout, but average the two highest normalized mel bands so the GRU structurally consumes 19 inputs instead of 20.

EVIDENCE: Removing a recurrent step failed even after increasing width to 59, indicating temporal context is currently more valuable than added capacity; the successful 58-unit, 25-step design motivates preserving its dynamics while testing redundancy between adjacent spectral bands.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)

    def initial_state(
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(19, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(174, 8)

    def _compress_input(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        high_band_average = normalized[..., 18:20].mean(dim=-1, keepdim=True)
        return torch.cat((normalized[..., :18], high_band_average), dim=-1)

    def initial_state(
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self._compress_input(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self._compress_input(frames), hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE