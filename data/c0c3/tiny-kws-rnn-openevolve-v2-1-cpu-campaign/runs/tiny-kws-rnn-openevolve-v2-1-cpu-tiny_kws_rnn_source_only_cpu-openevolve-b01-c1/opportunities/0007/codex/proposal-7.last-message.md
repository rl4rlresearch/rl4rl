MECHANISM: Boundary-seeking full-resolution GRU width compression

HYPOTHESIS: An 88-unit GRU processing all 32 frames will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 7.8% versus the verified 92-unit model.

INTENDED_EDIT: Reduce the GRU, recurrent state, temporal summary, and classifier width from 92 to 88 while preserving the full causal frame schedule and training procedure.

EVIDENCE: The 92-unit full-resolution GRU achieved 85.89% accuracy with 806,184,960 recurrent MACs, leaving enough observed accuracy margin to test the next four-unit structural reduction without discarding valuable frames.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 92, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(92, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 92, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 92, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 88, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(88, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 88, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 88, device=device, dtype=dtype)
>>>>>>> REPLACE