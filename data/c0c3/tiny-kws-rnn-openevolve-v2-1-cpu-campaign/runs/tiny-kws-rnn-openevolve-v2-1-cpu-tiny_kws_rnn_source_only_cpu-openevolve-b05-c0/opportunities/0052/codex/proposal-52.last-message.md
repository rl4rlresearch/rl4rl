MECHANISM: Learned-affine spectral pair pooling

HYPOTHESIS: Pooling the final two normalized mel bands before the GRU will retain at least 85% accuracy while preserving all 21 steps and reducing recurrent inference by exactly 2,978,010 MACs and parameters by 174.

INTENDED_EDIT: Keep the verified temporal schedule, hidden width, summaries, and classifier, but compress the two highest mel-band features into one elementwise mean after LayerNorm and change the GRU input width from 20 to 19.

EVIDENCE: The 114-feature, 21-step design meets the threshold at 85.03%, while reducing hidden width, recurrent steps, or classifier information failed; adjacent-band pooling tests a new structural cost axis while retaining aggregate information from every input band.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)
=======
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(19, 58, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(114, 8)

    def _recurrent_input(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        pooled_high_band = normalized[..., -2:].mean(dim=-1, keepdim=True)
        return torch.cat((normalized[..., :-2], pooled_high_band), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self._recurrent_input(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self._recurrent_input(frames), hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE