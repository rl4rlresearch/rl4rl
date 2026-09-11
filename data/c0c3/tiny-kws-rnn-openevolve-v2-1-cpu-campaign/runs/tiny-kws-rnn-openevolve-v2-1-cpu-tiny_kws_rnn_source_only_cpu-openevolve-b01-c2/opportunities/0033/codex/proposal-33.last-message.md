MECHANISM: Fixed adjacent-band spectral pooling

HYPOTHESIS: Averaging the two highest-frequency mel bands into one GRU input will preserve at least 85% validation accuracy while reducing total inference MACs to approximately 290,759,400.

INTENDED_EDIT: Keep all 20 bands in non-affine normalization, merge the adjacent upper-edge pair without learned operations, and reduce the GRU input width from 20 to 19.

EVIDENCE: Dropping either edge band reduced MACs to 290,759,400 but narrowly missed qualification; pooling preserves information from both bands while exploiting their expected local spectral redundancy.

<<<<<<< SEARCH
        self.gru = nn.GRU(
            20, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )

    def initial_state(
=======
        self.gru = nn.GRU(
            19, self.hidden_size, num_layers=1, batch_first=True
        )
        self.classifier = nn.Linear(
            3 * self.hidden_size, 7, bias=False
        )

    def compress_input(self, frames: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(frames)
        upper_pair = normalized[..., 18:].mean(dim=-1, keepdim=True)
        return torch.cat((normalized[..., :18], upper_pair), dim=-1)

    def initial_state(
>>>>>>> REPLACE

<<<<<<< SEARCH
        output, hidden = self.gru(
            self.input_norm(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
=======
        output, hidden = self.gru(
            self.compress_input(frame).unsqueeze(1),
            hidden.transpose(0, 1).contiguous(),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        outputs, hidden = self.gru(
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
        )
=======
        outputs, hidden = self.gru(
            self.compress_input(frames), hidden.transpose(0, 1).contiguous()
        )
>>>>>>> REPLACE