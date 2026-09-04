MECHANISM: Adjacent high-frequency mel-band pooling

HYPOTHESIS: Merging only the two highest adjacent mel bands will retain at least 85% validation accuracy while reducing recurrent MACs by 7,503,705 versus the verified 99-unit, 31-step design.

INTENDED_EDIT: Preserve the 99-unit state, 31-frame schedule, and full classifier while replacing the GRU’s 20-dimensional input with 19 features formed by retaining bands 0–17 and averaging bands 18–19.

EVIDENCE: The 99-unit, 31-step model met the threshold at 85.03%, whereas reducing hidden width to 98 or compressing the recurrent readout failed; this motivates preserving recurrent capacity and readout information while conservatively exploiting locality in the ordered mel inputs.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(19, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
=======
    def _input_features(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        high_band = normalized[..., 18:].mean(dim=-1, keepdim=True)
        return torch.cat((normalized[..., :18], high_band), dim=-1)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self._input_features(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self._input_features(frames), hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE