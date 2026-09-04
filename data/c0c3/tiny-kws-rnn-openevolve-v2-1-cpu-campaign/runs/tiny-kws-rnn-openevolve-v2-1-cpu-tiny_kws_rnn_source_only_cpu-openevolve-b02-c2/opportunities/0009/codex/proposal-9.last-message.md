MECHANISM: Full-resolution one-unit boundary refinement

HYPOTHESIS: A 99-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing total inference MACs by approximately 1.8% versus the qualified 100-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, temporal summary, and classifier width from 100 to 99 while preserving the full-frame schedule and established training procedure.

EVIDENCE: The 100-unit model qualified at 85.40% accuracy while the 98-unit model reached 84.54%; 99 units is the only untested integer midpoint at the observed width–accuracy boundary.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 100, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(100, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 100, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 100, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 99, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(99, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 99, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 99, device=device, dtype=dtype)
>>>>>>> REPLACE