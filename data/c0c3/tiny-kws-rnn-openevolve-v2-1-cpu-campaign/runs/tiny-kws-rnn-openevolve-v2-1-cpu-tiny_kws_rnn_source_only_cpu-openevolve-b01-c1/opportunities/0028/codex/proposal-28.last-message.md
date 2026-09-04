MECHANISM: Fixed adjacent-band spectral folding

HYPOTHESIS: Averaging the two highest mel bands into one feature will retain their combined information and recover at least 85% validation accuracy while matching the 19-band model’s approximately 699.7M total inference MACs.

INTENDED_EDIT: Structurally reduce the GRU input to 19 features while replacing high-band deletion with a fixed average of bands 18 and 19 in both recurrent execution paths.

EVIDENCE: Simply dropping the highest band narrowly missed the target at 84.91% while improving cross-entropy and reducing MACs; folding that band into its adjacent retained band preserves the cost reduction while addressing the likely information loss.

<<<<<<< SEARCH
        self.input_norm = nn.LayerNorm(20)
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)
=======
        self.input_norm = nn.LayerNorm(19)
        self.gru = nn.GRU(19, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)

    @staticmethod
    def _fold_bands(frames: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            (
                frames[..., :18],
                frames[..., 18:20].mean(dim=-1, keepdim=True),
            ),
            dim=-1,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frame).unsqueeze(1),
=======
            self.input_norm(self._fold_bands(frame)).unsqueeze(1),
>>>>>>> REPLACE

<<<<<<< SEARCH
            self.input_norm(frames), hidden.transpose(0, 1).contiguous()
=======
            self.input_norm(self._fold_bands(frames)),
            hidden.transpose(0, 1).contiguous(),
>>>>>>> REPLACE