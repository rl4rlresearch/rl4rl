MECHANISM: Fixed adjacent-band compression

HYPOTHESIS: Averaging the two highest adjacent mel bands into one feature will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 6.87 million (0.73%) versus the passing 117-unit, 24-step model.

INTENDED_EDIT: Reduce the GRU input from 20 to 19 features using parameter-free averaging of the two highest mel bands, while preserving recurrent width, schedule, readout, and training.

EVIDENCE: The 117-unit, 24-step design passed at 85.153%, while 116 hidden units and multiple 23-step schedules failed; a minimal compression of redundant adjacent frequency information explores a lower-risk structural cost axis.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)

    def initial_state(
=======
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)

    @staticmethod
    def _compress_bands(frames: torch.Tensor) -> torch.Tensor:
        upper_band_mean = frames[..., 18:20].mean(dim=-1, keepdim=True)
        return torch.cat((frames[..., :18], upper_band_mean), dim=-1)

    def initial_state(
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(self._compress_bands(frame)).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(self._compress_bands(frames)),
            hidden.transpose(0, 1).contiguous(),
>>>>>>> REPLACE