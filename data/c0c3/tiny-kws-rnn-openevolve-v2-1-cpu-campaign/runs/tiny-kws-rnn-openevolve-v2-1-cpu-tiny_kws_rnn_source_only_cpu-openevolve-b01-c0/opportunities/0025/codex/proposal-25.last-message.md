MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 119-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.54% versus the passing 120-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 120 to 119 while preserving the verified schedule and training procedure.

EVIDENCE: Every tested additive-readout width from 128 through 120 has passed, and the 120-unit model achieved 86.135% accuracy, providing a 1.135-point margin for the next isolated one-channel reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 119, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(119, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 119, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 119, device=device, dtype=dtype)
>>>>>>> REPLACE