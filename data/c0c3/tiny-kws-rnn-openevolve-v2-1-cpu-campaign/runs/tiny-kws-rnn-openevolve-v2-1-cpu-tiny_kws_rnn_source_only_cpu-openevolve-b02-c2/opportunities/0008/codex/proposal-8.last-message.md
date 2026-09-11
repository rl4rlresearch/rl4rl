MECHANISM: Full-resolution midpoint width reduction

HYPOTHESIS: A 98-unit GRU using all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by about 3.6% versus the qualified 100-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 120 to 98 while preserving the full-frame schedule and established training procedure.

EVIDENCE: The 100-unit model qualified at 85.40% accuracy while the 96-unit model reached 84.29%; 98 units is the most informative untested midpoint at the observed width–accuracy boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 120, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(120, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 120, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 120, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 98, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(98, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 98, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 98, device=device, dtype=dtype)
>>>>>>> REPLACE