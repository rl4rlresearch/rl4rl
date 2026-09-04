MECHANISM: One-channel recurrent-width trim

HYPOTHESIS: A 123-unit GRU with the verified 25-step additive mean–final readout will retain at least 85% validation accuracy while reducing exact dense inference MACs by approximately 1.5% versus the passing 124-unit model.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 124 to 123 while preserving the verified frame schedule and training procedure.

EVIDENCE: The 124-unit model achieved 86.012% accuracy, more than one percentage point above the requirement, and every successive additive-readout width from 128 through 124 has passed; a one-channel trim is therefore the smallest isolated structural cost reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 124, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(124, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 124, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 124, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 123, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(123, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 123, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 123, device=device, dtype=dtype)
>>>>>>> REPLACE