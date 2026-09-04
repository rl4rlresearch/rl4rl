MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 116-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs versus the passing 117-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 117 to 116 while preserving the verified schedule and training procedure.

EVIDENCE: Every tested additive-readout width from 128 through 117 has passed; the current 117-unit model achieved 85.767% accuracy, providing evidence for one more isolated channel reduction as the next test of the width boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 117, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(117, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 117, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 117, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 116, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(116, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 116, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 116, device=device, dtype=dtype)
>>>>>>> REPLACE