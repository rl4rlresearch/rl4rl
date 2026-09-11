MECHANISM: Full-resolution GRU width compression

HYPOTHESIS: A 104-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 12.8% versus the verified 112-unit design.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier width from 112 to 104 while preserving the full causal frame schedule and training procedure.

EVIDENCE: The 112-unit full-resolution GRU achieved 85.89% accuracy, while temporal subsampling failed at both 16 and 24 steps; this motivates preserving all 32 frames and testing the next structural width reduction.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 112, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(112, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 112, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 112, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(104, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
>>>>>>> REPLACE