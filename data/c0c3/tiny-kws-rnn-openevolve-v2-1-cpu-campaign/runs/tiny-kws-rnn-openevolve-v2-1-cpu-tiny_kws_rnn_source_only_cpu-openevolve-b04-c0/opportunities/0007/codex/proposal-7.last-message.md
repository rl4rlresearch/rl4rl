MECHANISM: Dual-readout recurrent width reduction

HYPOTHESIS: A 96-unit GRU with the proven 31-frame temporal-mean/final-state readout will retain at least 85% validation accuracy while reducing recurrent MACs by approximately 14% versus the verified 104-unit design.

INTENDED_EDIT: Reduce the GRU hidden state, online summary, and classifier input width from 104/208 to 96/192 while preserving the 31-frame schedule and training procedure.

EVIDENCE: The 104-unit, 31-step dual-readout model achieved 87.73% accuracy—2.73 points above the requirement—whereas the earlier mean-only 104-unit model achieved 85.03%; this margin supports testing a structural width reduction without discarding additional frames.

<<<<<<< SEARCH
        self.gru = nn.GRU(20, 104, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(208, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 104, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 104, device=device, dtype=dtype)
=======
        self.gru = nn.GRU(20, 96, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(192, 8)

    def initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, 1, 96, device=device, dtype=dtype)
        summary = torch.zeros(batch_size, 96, device=device, dtype=dtype)
>>>>>>> REPLACE